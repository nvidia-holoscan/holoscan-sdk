# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("run_rate_restart_conformance_probe.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("rate_restart_probe_runner", MODULE_PATH)
assert SPEC
assert SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def valid_properties() -> dict[str, object]:
    return {
        "restart_probe_kind": "instrumented_conformance",
        "restart_discover_callback_count": "2",
        "restart_allocate_callback_count": "2",
        "restart_request_to_discover_ns": "10",
        "restart_discover_to_user_allocate_ns": "20",
        "restart_user_allocate_to_successor_compute_ns": "30",
        "restart_request_to_successor_compute_ns": "60",
        "restart_initial_run_generation": "1",
        "restart_successor_run_generation": "1",
        "restart_initial_rate_generation": "1",
        "restart_successor_rate_generation": "2",
        "restart_published_run_generation": "1",
        "restart_published_rate_generation": "2",
        "restart_retained_allocation_generation": "1",
        "restart_previous_allocation_generation": "1",
        "restart_maximum_allocation_generation": "1",
        "restart_storage_binding_generation": "2",
        "restart_allocation_action": "reused_larger",
        "restart_materialization_count": "1",
        "restart_live_materialization_count": "1",
        "restart_peak_live_materialization_count": "1",
        "restart_physical_slot_count": "8",
        "restart_active_entry_limit": "4",
        "restart_live_physical_entry_count": "8",
        "restart_peak_live_physical_entry_count": "8",
        "restart_live_logical_requested_payload_bytes": "64",
        "restart_peak_live_logical_requested_payload_bytes": "64",
        "restart_logical_requested_payload_byte_span_per_slot": "8",
        "restart_provider_materialized_entries": "8",
        "restart_post_stop_live_materialization_count": "0",
        "restart_post_stop_live_physical_entry_count": "0",
        "restart_post_stop_live_logical_requested_payload_bytes": "0",
        "restart_provider_reserved_bytes_available": "false",
        "restart_provider_reserved_bytes_unavailable_reason": (
            "provider_does_not_publish_reservation_bytes"
        ),
        "restart_minimum_pool_generation": "1",
        "restart_maximum_pool_generation": "2",
        "restart_maximum_published_pool_generation": "2",
    }


def write_gtest_json(
    path: Path,
    properties: dict[str, object] | None = None,
    *,
    gtest_filter: str | None = None,
    test_name: str | None = None,
    failures: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_filter = gtest_filter or runner.PROBE_FILTER
    suite_name, selected_test_name = selected_filter.split(".", 1)
    test = {
        "name": test_name or selected_test_name,
        "status": "RUN",
        "result": "COMPLETED",
        "time": "0.001s",
        "classname": suite_name,
        **(properties or valid_properties()),
    }
    if failures:
        test["failures"] = [{"failure": "failed"}]
    path.write_text(
        json.dumps(
            {
                "name": "AllTests",
                "tests": 1,
                "failures": failures,
                "disabled": 0,
                "errors": 0,
                "testsuites": [
                    {
                        "name": suite_name,
                        "tests": 1,
                        "failures": failures,
                        "disabled": 0,
                        "errors": 0,
                        "testsuite": [test],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_checked_gtest_result_accepts_complete_probe(tmp_path: Path) -> None:
    path = tmp_path / "gtest.json"
    write_gtest_json(path)

    result = runner.checked_gtest_result(path)

    assert result["timings_ns"] == {
        "request_to_discover": 10,
        "discover_to_user_allocate": 20,
        "user_allocate_to_successor_compute": 30,
        "request_to_successor_compute": 60,
    }
    assert result["callbacks"] == {"discover": 2, "allocate": 2}
    resources = result["resources"]
    assert resources["allocation_action"] == "reused_larger"
    assert resources["retained_allocation_generation"] == 1
    assert resources["provider_materialized_entries"] == 8
    assert resources["logical_requested_payload_bytes"] == {
        "live": 64,
        "peak": 64,
        "byte_span_per_slot": 8,
        "scope": "requested_payload_only_excludes_provider_overhead",
    }
    assert resources["provider_reserved_bytes"]["available"] is False
    assert resources["provider_reserved_bytes"]["bytes"] is None


def test_checked_gtest_pass_accepts_each_exact_matrix_probe(tmp_path: Path) -> None:
    for scenario in runner.SCENARIO_PROBES:
        path = tmp_path / scenario.slug / "gtest.json"
        write_gtest_json(path, {}, gtest_filter=scenario.gtest_filter)

        result = runner.checked_gtest_pass(path, scenario.gtest_filter)

        assert result == {"assertion_outcome": "passed"}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("restart_request_to_successor_compute_ns", "61", "timing segments"),
        ("restart_discover_callback_count", "3", "callback cardinalities"),
        ("restart_successor_run_generation", "2", "generation facts"),
        ("restart_peak_live_materialization_count", "2", "resource invariants"),
    ],
)
def test_checked_gtest_result_rejects_inconsistent_probe(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    properties = valid_properties()
    properties[field] = value
    path = tmp_path / "gtest.json"
    write_gtest_json(path, properties)

    with pytest.raises(RuntimeError, match=message):
        runner.checked_gtest_result(path)


def test_checked_gtest_result_rejects_invented_provider_reserved_bytes(tmp_path: Path) -> None:
    properties = valid_properties()
    properties["restart_provider_reserved_bytes_total"] = "64"
    path = tmp_path / "gtest.json"
    write_gtest_json(path, properties)

    with pytest.raises(RuntimeError, match="reservation availability"):
        runner.checked_gtest_result(path)


def test_checked_gtest_result_rejects_wrong_test_identity(tmp_path: Path) -> None:
    path = tmp_path / "gtest.json"
    write_gtest_json(path, test_name="DifferentTest")

    with pytest.raises(RuntimeError, match="exact conformance probe"):
        runner.checked_gtest_result(path)


def test_checked_gtest_result_rejects_wrong_test_class(tmp_path: Path) -> None:
    path = tmp_path / "gtest.json"
    write_gtest_json(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["testsuites"][0]["testsuite"][0]["classname"] = "DifferentSuite"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(RuntimeError, match="exact conformance probe"):
        runner.checked_gtest_result(path)


@pytest.mark.parametrize("scope", ["top-level", "suite"])
def test_checked_gtest_result_rejects_wrong_container_name(tmp_path: Path, scope: str) -> None:
    path = tmp_path / "gtest.json"
    write_gtest_json(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    record = raw if scope == "top-level" else raw["testsuites"][0]
    record["name"] = "DifferentSuite"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(RuntimeError, match="exact conformance probe"):
        runner.checked_gtest_result(path)


@pytest.mark.parametrize(
    ("scope", "counter", "value"),
    [
        ("top-level", "tests", 0),
        ("top-level", "failures", 1),
        ("top-level", "disabled", 1),
        ("top-level", "errors", 1),
        ("suite", "tests", 0),
        ("suite", "failures", 1),
        ("suite", "disabled", 1),
        ("suite", "errors", 1),
    ],
)
def test_checked_gtest_result_rejects_incoherent_counters(
    tmp_path: Path, scope: str, counter: str, value: int
) -> None:
    path = tmp_path / "gtest.json"
    write_gtest_json(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    record = raw if scope == "top-level" else raw["testsuites"][0]
    record[counter] = value
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(RuntimeError, match=rf"{scope} {counter} counter"):
        runner.checked_gtest_result(path)


def test_checked_gtest_result_rejects_extra_suite(tmp_path: Path) -> None:
    path = tmp_path / "gtest.json"
    write_gtest_json(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["testsuites"].append(
        {
            "name": "EmptySuite",
            "tests": 0,
            "failures": 0,
            "disabled": 0,
            "errors": 0,
            "testsuite": [],
        }
    )
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(RuntimeError, match="exactly one test suite"):
        runner.checked_gtest_result(path)


def test_run_case_uses_exact_filter_and_preserves_raw_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "runtime_test"
    binary.write_bytes(b"binary")
    frozen_sha256 = runner.file_sha256(binary)
    output_dir = tmp_path / "evidence"
    observed_command: list[str] = []

    scenario = runner.SCENARIO_PROBES[0]

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        observed_command.extend(command)
        output_argument = next(item for item in command if item.startswith("--gtest_output=json:"))
        write_gtest_json(
            Path(output_argument.removeprefix("--gtest_output=json:")),
            {},
            gtest_filter=scenario.gtest_filter,
        )
        return subprocess.CompletedProcess(command, 0, stdout="probe stdout", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.run_case(binary, frozen_sha256, scenario, 3, output_dir, 5.0)

    assert f"--gtest_filter={scenario.gtest_filter}" in observed_command
    assert result["scenario"] == scenario.slug
    assert result["repetition"] == 3
    assert result["raw_gtest_json"] == f"repetition_003/{scenario.slug}/gtest.json"
    assert runner.valid_sha256(result["raw_gtest_json_sha256"])
    assert (output_dir / result["raw_gtest_json"]).is_file()
    assert json.loads((output_dir / f"repetition_003/{scenario.slug}/command.json").read_text())[
        1
    ] == (f"--gtest_filter={scenario.gtest_filter}")


def test_run_case_rejects_stale_gtest_json_when_process_writes_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "runtime_test"
    binary.write_bytes(b"binary")
    frozen_sha256 = runner.file_sha256(binary)
    output_dir = tmp_path / "evidence"
    scenario = runner.SCENARIO_PROBES[0]
    stale_result = output_dir / "repetition_003" / scenario.slug / "gtest.json"
    write_gtest_json(stale_result, {}, gtest_filter=scenario.gtest_filter)

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="probe stdout", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="no fresh gtest JSON"):
        runner.run_case(binary, frozen_sha256, scenario, 3, output_dir, 5.0)
    assert not stale_result.exists()


def test_run_hsb_case_preserves_exact_assertion_command_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "multi_tensor_schema_runtime_validation"
    binary.write_bytes(b"hsb-binary")
    frozen_sha256 = runner.file_sha256(binary)
    output_dir = tmp_path / "evidence"
    scenario = runner.HSB_SCENARIO_PROBES[0]
    observed_command: list[str] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        observed_command.extend(command)
        return subprocess.CompletedProcess(command, 0, stdout="assertion passed", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.run_hsb_case(binary, frozen_sha256, scenario, 2, output_dir, 5.0)

    assert observed_command == [str(binary), scenario.argument]
    assert result["scenario"] == scenario.slug
    assert result["assertion_outcome"] == "passed"
    assert result["command"] == f"hsb_shaped/repetition_002/{scenario.slug}/command.json"
    assert runner.valid_sha256(result["stdout_sha256"])
    assert (output_dir / result["stdout"]).read_text() == "assertion passed"


def test_run_case_rejects_runtime_binary_drift_after_each_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "runtime_test"
    binary.write_bytes(b"before")
    frozen_sha256 = runner.file_sha256(binary)
    scenario = runner.SCENARIO_PROBES[0]

    def drift_after_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        output_argument = next(item for item in command if item.startswith("--gtest_output=json:"))
        write_gtest_json(
            Path(output_argument.removeprefix("--gtest_output=json:")),
            {},
            gtest_filter=scenario.gtest_filter,
        )
        binary.write_bytes(b"after")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", drift_after_run)
    with pytest.raises(RuntimeError, match="runtime_test binary changed"):
        runner.run_case(binary, frozen_sha256, scenario, 1, tmp_path / "evidence", 5.0)


def test_run_hsb_case_rejects_binary_drift_after_each_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "multi_tensor_schema_runtime_validation"
    binary.write_bytes(b"before")
    frozen_sha256 = runner.file_sha256(binary)
    scenario = runner.HSB_SCENARIO_PROBES[0]

    def drift_after_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        binary.write_bytes(b"after")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", drift_after_run)
    with pytest.raises(RuntimeError, match="HSB validation binary changed"):
        runner.run_hsb_case(binary, frozen_sha256, scenario, 1, tmp_path / "evidence", 5.0)


def test_make_summary_has_source_options_and_no_latency_threshold(tmp_path: Path) -> None:
    binary = tmp_path / "runtime_test"
    binary.write_bytes(b"binary")
    path = tmp_path / "gtest.json"
    write_gtest_json(path)
    run = runner.checked_gtest_result(path)
    run.update(
        {
            "scenario": "smaller_compatible",
            "repetition": 1,
            "raw_gtest_json": "repetition_001/gtest.json",
            "raw_gtest_json_sha256": "b" * 64,
        }
    )
    args = argparse.Namespace(
        repetitions=1,
        process_timeout_sec=12.5,
        git_revision="abc123",
        source_state="dirty-frozen-patch",
        source_snapshot_sha256="a" * 64,
    )

    matrix_runs = []
    for scenario in runner.SCENARIO_PROBES:
        if scenario.slug == "smaller_compatible":
            matrix_runs.append(run)
        else:
            matrix_runs.append(
                {
                    "scenario": scenario.slug,
                    "repetition": 1,
                    "raw_gtest_json": f"repetition_001/{scenario.slug}/gtest.json",
                    "raw_gtest_json_sha256": "b" * 64,
                    "assertion_outcome": "passed",
                }
            )

    frozen_binary_sha256 = "c" * 64
    summary = runner.make_summary(args, binary, matrix_runs, frozen_binary_sha256)

    assert summary["probe"] == {
        "gtest_filters": [scenario.gtest_filter for scenario in runner.SCENARIO_PROBES],
        "repetitions": 1,
        "process_timeout_sec": 12.5,
    }
    assert summary["source"]["git_revision"] == "abc123"
    assert summary["source"]["source_snapshot_sha256"] == "a" * 64
    assert summary["binary"]["sha256"] == frozen_binary_sha256
    assert summary["qualification"]["product_latency_threshold"] is None
    assert summary["callback_observations"] == [{"discover": 2, "allocate": 2}]
    assert [row["scenario"] for row in summary["restart_matrix"]] == [
        scenario.slug for scenario in runner.SCENARIO_PROBES
    ]
    assert all(row["passing_repetitions"] == 1 for row in summary["restart_matrix"])
    assert len(summary["restart_matrix"]) == 6
    assert summary["restart_matrix"][1]["measurement_scope"] == "instrumented_properties"
    assert summary["restart_matrix"][0]["measurement_scope"] == "focused_gtest_assertions"
    assert summary["restart_matrix"][0]["timings_ns"] is None
    assert summary["restart_matrix"][0]["timings_unavailable_reason"] == (
        "focused_gtest_does_not_record_timing_properties"
    )
    replacement = next(
        row
        for row in summary["restart_matrix"]
        if row["scenario"] == "synthetic_incompatible_replacement"
    )
    assert replacement["evidence_boundary"] == (
        "synthetic_rate_reuse_compatibility_fault_not_actual_layout"
    )
    assert summary["hsb_shaped_boundary"]["status"] == "not_collected"
    assert summary["hsb_shaped_boundary"]["remaining_gap"] == (
        "actual_hsb_camera_provider_matrix_with_timing_and_allocator_registration_counters"
    )
    assert summary["timings_ns"]["request_to_successor_compute"] == {
        "min": 60,
        "median": 60,
        "p95": 60,
        "max": 60,
    }


def test_make_summary_records_hsb_shaped_assertions_without_fabricated_measurements(
    tmp_path: Path,
) -> None:
    runtime_binary = tmp_path / "runtime_test"
    runtime_binary.write_bytes(b"runtime")
    hsb_binary = tmp_path / "multi_tensor_schema_runtime_validation"
    hsb_binary.write_bytes(b"hsb")
    path = tmp_path / "gtest.json"
    write_gtest_json(path)
    instrumented = runner.checked_gtest_result(path)
    matrix_runs = []
    for scenario in runner.SCENARIO_PROBES:
        row = {
            "scenario": scenario.slug,
            "repetition": 1,
            "raw_gtest_json": f"repetition_001/{scenario.slug}/gtest.json",
            "raw_gtest_json_sha256": "b" * 64,
            "assertion_outcome": "passed",
        }
        if scenario.instrumented:
            row.update(instrumented)
        matrix_runs.append(row)
    hsb_runs = [
        {
            "scenario": scenario.slug,
            "repetition": 1,
            "assertion_outcome": "passed",
            "command": f"hsb_shaped/repetition_001/{scenario.slug}/command.json",
            "stdout": f"hsb_shaped/repetition_001/{scenario.slug}/stdout.log",
            "stdout_sha256": "c" * 64,
            "stderr": f"hsb_shaped/repetition_001/{scenario.slug}/stderr.log",
            "stderr_sha256": "d" * 64,
        }
        for scenario in runner.HSB_SCENARIO_PROBES
    ]
    args = argparse.Namespace(
        repetitions=1,
        process_timeout_sec=12.5,
        git_revision="abc123",
        source_state="dirty-frozen-patch",
        source_snapshot_sha256="a" * 64,
        hsb_validation_binary=hsb_binary,
    )

    frozen_runtime_sha256 = "e" * 64
    frozen_hsb_sha256 = "f" * 64
    summary = runner.make_summary(
        args,
        runtime_binary,
        matrix_runs,
        frozen_runtime_sha256,
        hsb_binary,
        hsb_runs,
        frozen_hsb_sha256,
    )

    boundary = summary["hsb_shaped_boundary"]
    assert summary["binary"]["sha256"] == frozen_runtime_sha256
    assert boundary["status"] == "assertion_backed_partial_coverage"
    assert boundary["binary"]["sha256"] == frozen_hsb_sha256
    assert len(boundary["scenarios"]) == 2
    assert all(row["timings_ns"] is None for row in boundary["scenarios"])
    assert all(row["allocator_call_count"] is None for row in boundary["scenarios"])
    assert all(row["registration_call_count"] is None for row in boundary["scenarios"])


def test_main_rejects_runtime_binary_mutation_between_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    binary = tmp_path / "runtime_test"
    binary.write_bytes(b"original-binary")
    output_dir = tmp_path / "evidence"
    args = argparse.Namespace(
        runtime_test_binary=binary,
        repetitions=1,
        output_dir=output_dir,
        process_timeout_sec=12.5,
        git_revision="abc123",
        source_state="frozen-source",
        source_snapshot_sha256="a" * 64,
        hsb_validation_binary=None,
    )
    calls = 0

    def fake_run_case(
        _: Path,
        __: str,
        scenario: runner.ScenarioProbe,
        repetition: int,
        ___: Path,
        ____: float,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            binary.write_bytes(b"mutated-between-runs")
        return {
            "scenario": scenario.slug,
            "repetition": repetition,
            "raw_gtest_json": f"repetition_001/{scenario.slug}/gtest.json",
            "raw_gtest_json_sha256": "b" * 64,
            "assertion_outcome": "passed",
        }

    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "run_case", fake_run_case)

    assert runner.main() == 2
    assert calls == len(runner.SCENARIO_PROBES)
    assert not (output_dir / "summary.json").exists()
    assert (
        "runtime_test binary changed while collecting restart evidence" in capsys.readouterr().err
    )


def test_main_rejects_hsb_binary_mutation_between_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime_binary = tmp_path / "runtime_test"
    runtime_binary.write_bytes(b"runtime-binary")
    hsb_binary = tmp_path / "multi_tensor_schema_runtime_validation"
    hsb_binary.write_bytes(b"original-hsb-binary")
    output_dir = tmp_path / "evidence"
    args = argparse.Namespace(
        runtime_test_binary=runtime_binary,
        repetitions=1,
        output_dir=output_dir,
        process_timeout_sec=12.5,
        git_revision="abc123",
        source_state="frozen-source",
        source_snapshot_sha256="a" * 64,
        hsb_validation_binary=hsb_binary,
    )
    hsb_calls = 0

    def fake_run_case(
        _: Path,
        __: str,
        scenario: runner.ScenarioProbe,
        repetition: int,
        ___: Path,
        ____: float,
    ) -> dict[str, object]:
        return {
            "scenario": scenario.slug,
            "repetition": repetition,
            "raw_gtest_json": f"repetition_001/{scenario.slug}/gtest.json",
            "raw_gtest_json_sha256": "b" * 64,
            "assertion_outcome": "passed",
        }

    def fake_run_hsb_case(
        _: Path,
        __: str,
        scenario: runner.HsbScenarioProbe,
        repetition: int,
        ___: Path,
        ____: float,
    ) -> dict[str, object]:
        nonlocal hsb_calls
        hsb_calls += 1
        if hsb_calls == 1:
            hsb_binary.write_bytes(b"mutated-between-hsb-runs")
        return {
            "scenario": scenario.slug,
            "repetition": repetition,
            "assertion_outcome": "passed",
        }

    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "run_case", fake_run_case)
    monkeypatch.setattr(runner, "run_hsb_case", fake_run_hsb_case)

    assert runner.main() == 2
    assert hsb_calls == len(runner.HSB_SCENARIO_PROBES)
    assert not (output_dir / "summary.json").exists()
    assert (
        "HSB validation binary changed while collecting restart evidence" in capsys.readouterr().err
    )


def test_validate_args_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="positive"):
        runner.validate_args(
            argparse.Namespace(
                repetitions=0,
                process_timeout_sec=1.0,
                source_snapshot_sha256=None,
            )
        )
    with pytest.raises(ValueError, match="SHA-256"):
        runner.validate_args(
            argparse.Namespace(
                repetitions=1,
                process_timeout_sec=1.0,
                source_snapshot_sha256="ABC",
            )
        )
