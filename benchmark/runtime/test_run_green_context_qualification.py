# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_green_context_qualification as qualification


def make_args(**updates: object) -> SimpleNamespace:
    values = {
        "benchmark_binary": Path("/tmp/runtime_graph_benchmark"),
        "output_dir": Path("/tmp/green-context-qualification"),
        "revision": "0123456789abcdef",
        "samples": 10_000,
        "warmup": 1_000,
        "repetitions": 5,
        "tensor_bytes": 1 << 20,
        "tensor_device": 0,
        "publisher_period_us": 1_000,
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
        ({"warmup": -1}, "--warmup must be nonnegative"),
        ({"repetitions": 1}, "--repetitions must be at least two"),
        ({"tensor_bytes": 0}, "--tensor-bytes must be positive"),
        ({"tensor_device": -1}, "--tensor-device must be nonnegative"),
        ({"publisher_period_us": 0}, "--publisher-period-us must be positive"),
        ({"process_timeout_sec": 0}, "qualification timeouts must be positive"),
        ({"revision": ""}, "--revision must be nonempty"),
    ],
)
def test_invalid_qualification_geometry_is_rejected(
    updates: dict[str, object], message: str
) -> None:
    # Key setup: vary one bounded matrix input while retaining every other valid default.
    args = make_args(**updates)

    # Key assertion: invalid geometry fails before any native benchmark process starts.
    with pytest.raises(ValueError, match=message):
        qualification.validate_args(args)


def test_matrix_has_matched_bound_and_unbound_topologies() -> None:
    # Key setup: inspect the immutable qualification cells rather than running CUDA.
    by_name = {cell.name: cell for cell in qualification.MATRIX}

    # Key assertion: each bound layout names an unbound baseline with identical lane topology.
    for cell in qualification.MATRIX:
        if cell.baseline is not None:
            assert by_name[cell.baseline].operator_lanes == cell.operator_lanes
            assert by_name[cell.baseline].layout == "none"


def test_realtime_disposition_matches_hot_path_scope() -> None:
    # Key setup: inspect the retained qualification-summary policy text.

    # Key assertion: RealTime is admitted without claiming bounded terminal CUDA quiescence.
    assert qualification.REALTIME_PROFILE_DISPOSITION == (
        "admitted: preallocated hot-path contract; terminal stream quiescence remains potentially "
        "unbounded"
    )


def test_instrumented_command_is_separate_from_performance_command() -> None:
    # Key setup: build both command forms for one dedicated-to-residual latency cell.
    args = make_args()
    cell = next(cell for cell in qualification.MATRIX if cell.name == "dedicated_to_residual")
    performance = qualification.native_command(
        args,
        cell,
        "latency",
        Path("/tmp/performance.json"),
        Path("/tmp/performance-plan.json"),
        False,
    )
    instrumented = qualification.native_command(
        args,
        cell,
        "latency",
        Path("/tmp/instrumentation.json"),
        Path("/tmp/instrumentation-plan.json"),
        True,
    )

    # Key assertion: performance samples omit source-tree counters while the diagnostic sample
    # records Runtime phases, CUDA events/waits, and an exact separate plan artifact.
    assert "--runtime-phase-counters" not in performance
    assert "--cuda-resource-counters" not in performance
    assert "--runtime-phase-counters" in instrumented
    assert "--cuda-resource-counters" in instrumented
    assert performance[performance.index("--green-context-layout") + 1] == ("dedicated_to_residual")
    assert performance[performance.index("--plan-output") + 1] == "/tmp/performance-plan.json"

    saturated = qualification.native_command(
        args,
        cell,
        "saturated",
        Path("/tmp/saturated.json"),
        Path("/tmp/saturated-plan.json"),
        True,
    )
    assert "--cuda-resource-counters" in saturated
    assert "--runtime-phase-counters" not in saturated


def test_checked_result_requires_p95_and_exact_cuda_retirement(tmp_path: Path) -> None:
    # Key setup: construct the smallest accepted instrumented result for one bound latency cell.
    cell = next(cell for cell in qualification.MATRIX if cell.name == "same_partition")
    path = tmp_path / "result.json"
    result = {
        "scenario": "tensor",
        "workload": "latency",
        "green_context": {"layout": "same_partition", "enabled": True},
        "validation": {"passed": True},
        "session": {"termination": "stopped", "diagnostics": []},
        "allocation_probe": {
            "source_compute_allocations": 0,
            "sink_compute_allocations": 0,
        },
        "cuda_resource_counters": {
            "enabled": True,
            "streams_created": 2,
            "streams_destroyed": 2,
            "events_created": 9,
            "events_destroyed": 9,
            "stream_quiesce_ns": 123,
        },
        "runtime_thread_usage": {"enabled": True, "aggregate": {"samples": 3}},
        "timing": {"latency_ns": {"p50": 10, "p95": 20, "p99": 30}},
        "cold_timing_ns": {"compile": 1},
    }
    path.write_text(json.dumps(result), encoding="utf-8")

    # Key assertion: exact event/stream retirement and p95 evidence admit the result, while
    # dropping p95 makes it unusable for Slice 11 reporting.
    assert qualification.checked_result(path, cell, "latency", True) == result
    del result["timing"]["latency_ns"]["p95"]
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="lacks p50/p95/p99"):
        qualification.checked_result(path, cell, "latency", True)


def test_checked_result_requires_runtime_cpu_usage(tmp_path: Path) -> None:
    # Key setup: construct an otherwise-valid instrumented result without a CPU sample.
    cell = next(cell for cell in qualification.MATRIX if cell.name == "same_partition")
    path = tmp_path / "result.json"
    result = {
        "scenario": "tensor",
        "workload": "latency",
        "green_context": {"layout": "same_partition", "enabled": True},
        "validation": {"passed": True},
        "session": {"termination": "stopped", "diagnostics": []},
        "allocation_probe": {
            "source_compute_allocations": 0,
            "sink_compute_allocations": 0,
        },
        "cuda_resource_counters": {
            "enabled": True,
            "streams_created": 2,
            "streams_destroyed": 2,
            "events_created": 9,
            "events_destroyed": 9,
            "stream_quiesce_ns": 123,
        },
        "runtime_thread_usage": {"enabled": True, "aggregate": {"samples": 0}},
        "timing": {"latency_ns": {"p50": 10, "p95": 20, "p99": 30}},
        "cold_timing_ns": {"compile": 1},
    }
    path.write_text(json.dumps(result), encoding="utf-8")

    # Key assertion: the retained instrumented row cannot claim CPU coverage without a successful
    # source-tree per-thread usage sample.
    with pytest.raises(RuntimeError, match="lacks Runtime CPU usage samples"):
        qualification.checked_result(path, cell, "latency", True)


def test_comparison_sign_is_favorable_for_lower_latency_and_higher_throughput() -> None:
    # Key setup: compare a ten-percent improvement in each metric direction.

    # Key assertion: both improvements use a positive favorable-delta convention.
    assert qualification.compare(90.0, 100.0, "latency") == pytest.approx(10.0)
    assert qualification.compare(110.0, 100.0, "saturated") == pytest.approx(10.0)


def test_nonempty_output_is_rejected_without_modification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Key setup: point a new invocation at retained evidence from a prior successful attempt.
    binary = tmp_path / "runtime_graph_benchmark"
    binary.touch()
    output_dir = tmp_path / "qualification"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text('{"revision":"stale"}\n', encoding="utf-8")
    (output_dir / "orphaned-run.json").write_text("{}\n", encoding="utf-8")
    args = make_args(benchmark_binary=binary, output_dir=output_dir, revision="current")
    monkeypatch.setattr(qualification, "parse_args", lambda: args)

    # Key assertion: the runner refuses the nonempty directory without changing prior evidence.
    with pytest.raises(ValueError, match="absent or empty"):
        qualification.main()
    assert (output_dir / "summary.json").read_text(encoding="utf-8") == ('{"revision":"stale"}\n')
    assert (output_dir / "orphaned-run.json").is_file()
    assert not (output_dir / "attempt.json").exists()


def test_symlink_output_is_rejected_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Key setup: point --output-dir through a symlink at unrelated retained data.
    binary = tmp_path / "runtime_graph_benchmark"
    binary.touch()
    target = tmp_path / "retained"
    target.mkdir()
    retained = target / "evidence.json"
    retained.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "qualification-link"
    output_dir.symlink_to(target, target_is_directory=True)
    args = make_args(benchmark_binary=binary, output_dir=output_dir, revision="current")
    monkeypatch.setattr(qualification, "parse_args", lambda: args)

    # Key assertion: no rename or cleanup operation follows the link or modifies its target.
    with pytest.raises(ValueError, match="symbolic link"):
        qualification.main()
    assert retained.read_text(encoding="utf-8") == "{}\n"
    assert output_dir.is_symlink()


def test_symlinked_output_parent_is_rejected_without_creating_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Key setup: place a missing output directory below an existing symlinked parent.
    binary = tmp_path / "runtime_graph_benchmark"
    binary.touch()
    target = tmp_path / "retained"
    target.mkdir()
    parent = tmp_path / "qualification-link"
    parent.symlink_to(target, target_is_directory=True)
    output_dir = parent / "nested-attempt"
    args = make_args(benchmark_binary=binary, output_dir=output_dir, revision="current")
    monkeypatch.setattr(qualification, "parse_args", lambda: args)

    # Key assertion: validation rejects the symlinked ancestor before mkdir can touch its target.
    with pytest.raises(ValueError, match="symbolic link"):
        qualification.main()
    assert not (target / "nested-attempt").exists()


def test_replaced_output_path_cannot_redirect_terminal_publication(tmp_path: Path) -> None:
    # Key setup: claim one directory, move it aside, and replace its path with a symlink to
    # unrelated retained evidence before terminal publication.
    output_dir = tmp_path / "qualification"
    attempt = qualification.QualificationAttempt.begin(output_dir, "current")
    claimed_dir = tmp_path / "claimed-qualification"
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external_summary = external_dir / "summary.json"
    external_summary.write_text('{"retained":true}\n', encoding="utf-8")
    output_dir.rename(claimed_dir)
    output_dir.symlink_to(external_dir, target_is_directory=True)

    try:
        # Key assertion: identity verification rejects the replaced path before handle-relative
        # cleanup or status publication, leaving both the claimed and external evidence intact.
        with pytest.raises(RuntimeError, match="directory identity changed"):
            attempt.finish("failed")
    finally:
        attempt.close()
    assert external_summary.read_text(encoding="utf-8") == '{"retained":true}\n'
    assert not (external_dir / "attempt.json").exists()
    retained_attempt = json.loads((claimed_dir / "attempt.json").read_text(encoding="utf-8"))
    assert retained_attempt["status"] == "running"


def test_empty_output_records_skipped_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Key setup: reserve an empty destination and report the documented unavailable status.
    binary = tmp_path / "runtime_graph_benchmark"
    binary.touch()
    output_dir = tmp_path / "qualification"
    output_dir.mkdir()
    args = make_args(benchmark_binary=binary, output_dir=output_dir, revision="current")
    monkeypatch.setattr(qualification, "parse_args", lambda: args)
    monkeypatch.setattr(qualification, "run_qualification", lambda _args: qualification.SKIPPED)

    # Key assertion: the owned directory contains an explicit terminal status but no summary.
    assert qualification.main() == qualification.SKIPPED
    attempt = json.loads((output_dir / "attempt.json").read_text(encoding="utf-8"))
    assert attempt["status"] == "skipped"
    assert attempt["revision"] == "current"
    assert not (output_dir / "summary.json").exists()


def test_successful_attempt_publishes_summary_and_terminal_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Key setup: model one successful qualification in a previously absent output directory.
    binary = tmp_path / "runtime_graph_benchmark"
    binary.touch()
    output_dir = tmp_path / "qualification"
    args = make_args(benchmark_binary=binary, output_dir=output_dir, revision="current")
    monkeypatch.setattr(qualification, "parse_args", lambda: args)

    def succeed(current_args: object) -> int:
        (current_args.output_dir / "summary.json").write_text("{}\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(qualification, "run_qualification", succeed)

    # Key assertion: success keeps the current summary and atomically publishes succeeded status.
    assert qualification.main() == 0
    attempt = json.loads((output_dir / "attempt.json").read_text(encoding="utf-8"))
    assert attempt["status"] == "succeeded"
    assert attempt["summary"] == "summary.json"


def test_terminal_status_failure_preserves_qualification_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Key setup: fail the native run and independently fail the terminal status publication.
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

    # Key assertion: automation observes the original qualification failure, and the last complete
    # status remains parseable instead of being truncated by the failed terminal update.
    with pytest.raises(RuntimeError, match="injected qualification failure"):
        qualification.main()
    attempt = json.loads((output_dir / "attempt.json").read_text(encoding="utf-8"))
    assert attempt["status"] == "running"
