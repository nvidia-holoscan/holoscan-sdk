#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect isolated Task-041 restart conformance probes and retain raw gtest evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import rate_evidence_utils as evidence

SCHEMA = "holoscan.ea2.rate-restart-conformance-probe/v2"
PROBE_FILTER = (
    "RunSessionTest.DiscoverRateRestartReusesCompatibleTensorBackingAndPublishesNewBinding"
)


class ScenarioProbe(NamedTuple):
    slug: str
    gtest_filter: str
    assertion_contract: str
    instrumented: bool = False
    evidence_boundary: str | None = None


SCENARIO_PROBES = (
    ScenarioProbe(
        "same_compatible",
        "RunSessionTest.SameDiscoverRateRestartReusesExactTensorBackingWithoutOverlap",
        "same selected rate reuses exact backing with no successor materialization",
    ),
    ScenarioProbe(
        "smaller_compatible",
        PROBE_FILTER,
        "smaller selected rate reuses larger backing and publishes a smaller active depth",
        True,
    ),
    ScenarioProbe(
        "larger_within_admitted",
        "RunSessionTest.MaximumBackedTensorRateGrowthWithinAdmittedCeilingReusesPhysicalBacking",
        "larger selected rate remains within the admitted maximum and reuses physical backing",
    ),
    ScenarioProbe(
        "synthetic_incompatible_replacement",
        "RunSessionTest.FiniteMaximumRateRestartMaterializesAdmittedReplacementWithBoundedOverlap",
        "synthetic compatibility fault forces admitted replacement with bounded "
        "predecessor-candidate overlap",
        False,
        "synthetic_rate_reuse_compatibility_fault_not_actual_layout",
    ),
    ScenarioProbe(
        "late_stage_incompatible_fail_closed",
        "RunSessionTest.ArmRateRestartNeverAllocatesReplacementBacking",
        "synthetic incompatible retained backing at kArm fails closed without "
        "replacement allocation",
        False,
        "supplemental_late_stage_assertion_not_the_incompatible_replacement_matrix_row",
    ),
    ScenarioProbe(
        "above_admitted_maximum",
        "RunSessionTest.AboveMaximumDiscoverRateRestartFailsBeforeAllocateAndRetainsPublishedBacking",
        "selected rate above the admitted maximum fails before restart allocation",
    ),
)


class HsbScenarioProbe(NamedTuple):
    slug: str
    argument: str
    assertion_contract: str


HSB_SCENARIO_PROBES = (
    HsbScenarioProbe(
        "compatible_smaller_rate_bundle",
        "rate-fields-restart",
        "three heterogeneous fields atomically reuse their finite-maximum backing "
        "at a smaller rate",
    ),
    HsbScenarioProbe(
        "atomic_publication_rejection",
        "rate-fields-restart-reject",
        "three-field candidate publication rejection leaves the predecessor bundle published",
    ),
)
TIMING_PROPERTIES = {
    "request_to_discover": "restart_request_to_discover_ns",
    "discover_to_user_allocate": "restart_discover_to_user_allocate_ns",
    "user_allocate_to_successor_compute": "restart_user_allocate_to_successor_compute_ns",
    "request_to_successor_compute": "restart_request_to_successor_compute_ns",
}
INTEGER_PROPERTIES = (
    "restart_discover_callback_count",
    "restart_allocate_callback_count",
    "restart_initial_run_generation",
    "restart_successor_run_generation",
    "restart_initial_rate_generation",
    "restart_successor_rate_generation",
    "restart_published_run_generation",
    "restart_published_rate_generation",
    "restart_retained_allocation_generation",
    "restart_previous_allocation_generation",
    "restart_maximum_allocation_generation",
    "restart_storage_binding_generation",
    "restart_materialization_count",
    "restart_live_materialization_count",
    "restart_peak_live_materialization_count",
    "restart_physical_slot_count",
    "restart_active_entry_limit",
    "restart_live_physical_entry_count",
    "restart_peak_live_physical_entry_count",
    "restart_live_logical_requested_payload_bytes",
    "restart_peak_live_logical_requested_payload_bytes",
    "restart_logical_requested_payload_byte_span_per_slot",
    "restart_provider_materialized_entries",
    "restart_post_stop_live_materialization_count",
    "restart_post_stop_live_physical_entry_count",
    "restart_post_stop_live_logical_requested_payload_bytes",
    "restart_minimum_pool_generation",
    "restart_maximum_pool_generation",
    "restart_maximum_published_pool_generation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_test_binary", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--process-timeout-sec", type=float, default=30.0)
    parser.add_argument("--git-revision", default=os.environ.get("HOLOSCAN_GIT_REVISION"))
    parser.add_argument("--source-state", default="unknown")
    parser.add_argument("--source-snapshot-sha256")
    parser.add_argument(
        "--hsb-validation-binary",
        type=Path,
        help="optional generated multi-Tensor validation binary for HSB-shaped boundary assertions",
    )
    return parser.parse_args()


def valid_sha256(value: object) -> bool:
    return evidence.valid_sha256(value)


def validate_args(args: argparse.Namespace) -> None:
    if args.repetitions <= 0 or args.process_timeout_sec <= 0:
        raise ValueError("repetitions and process timeout must be positive")
    if args.source_snapshot_sha256 is not None and not valid_sha256(args.source_snapshot_sha256):
        raise ValueError("--source-snapshot-sha256 must be a lowercase SHA-256 digest")


def file_sha256(path: Path) -> str:
    return evidence.file_sha256(path)


def require_unchanged_binary(path: Path, frozen_sha256: str, label: str) -> None:
    evidence.require_file_sha256(
        path,
        frozen_sha256,
        f"{label} binary changed while collecting restart evidence",
    )


def percentile(values: list[int], quantile: float) -> int:
    return int(evidence.nearest_rank_percentile(values, quantile))


def summarize(values: list[int]) -> dict[str, int | float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def build_metadata(binary: Path) -> dict[str, object]:
    return evidence.cmake_build_metadata(binary, include_compiler_version_output=False)


def _require_counter(
    record: dict[str, object], key: str, expected: int, scope: str, path: Path
) -> None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise RuntimeError(f"gtest JSON {scope} {key} counter must equal {expected}: {path}")


def _single_test(
    raw: dict[str, object], path: Path, gtest_filter: str = PROBE_FILTER
) -> dict[str, object]:
    suites = raw.get("testsuites")
    if not isinstance(suites, list) or len(suites) != 1 or not isinstance(suites[0], dict):
        raise RuntimeError(f"gtest JSON must contain exactly one test suite: {path}")
    suite = suites[0]
    cases = suite.get("testsuite")
    if not isinstance(cases, list) or len(cases) != 1 or not isinstance(cases[0], dict):
        raise RuntimeError(f"gtest JSON must contain exactly one test: {path}")
    test = cases[0]

    for key, expected in (("tests", 1), ("failures", 0), ("disabled", 0), ("errors", 0)):
        _require_counter(raw, key, expected, "top-level", path)
        _require_counter(suite, key, expected, "suite", path)

    test_failures = test.get("failures", [])
    suite_name, test_name = gtest_filter.split(".", 1)
    if (
        raw.get("name") != "AllTests"
        or suite.get("name") != suite_name
        or test.get("name") != test_name
        or test.get("classname") != suite_name
        or test.get("status") != "RUN"
        or test.get("result") != "COMPLETED"
        or not isinstance(test_failures, list)
        or test_failures
    ):
        raise RuntimeError(f"gtest JSON is not the passing exact conformance probe: {path}")
    return test


def checked_gtest_pass(path: Path, gtest_filter: str) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"gtest JSON root must be an object: {path}")
    _single_test(raw, path, gtest_filter)
    return {"assertion_outcome": "passed"}


def _properties(test: dict[str, object]) -> dict[str, object]:
    reserved = {"name", "status", "result", "timestamp", "time", "classname", "failures"}
    result = {key: value for key, value in test.items() if key not in reserved}
    nested = test.get("properties")
    if isinstance(nested, dict):
        result.update(nested)
    elif isinstance(nested, list):
        for item in nested:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                result[item["name"]] = item.get("value")
    return result


def _nonnegative_integer(properties: dict[str, object], key: str) -> int:
    value = properties.get(key)
    if isinstance(value, bool):
        raise RuntimeError(f"{key} must be a non-negative integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isdecimal():
        parsed = int(value)
    else:
        raise RuntimeError(f"{key} must be a non-negative integer")
    if parsed < 0:
        raise RuntimeError(f"{key} must be a non-negative integer")
    return parsed


def checked_gtest_result(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"gtest JSON root must be an object: {path}")
    properties = _properties(_single_test(raw, path))
    if properties.get("restart_probe_kind") != "instrumented_conformance":
        raise RuntimeError(f"gtest JSON lacks the instrumented conformance marker: {path}")

    timings = {
        name: _nonnegative_integer(properties, property_name)
        for name, property_name in TIMING_PROPERTIES.items()
    }
    if timings["request_to_successor_compute"] <= 0 or timings["request_to_successor_compute"] != (
        timings["request_to_discover"]
        + timings["discover_to_user_allocate"]
        + timings["user_allocate_to_successor_compute"]
    ):
        raise RuntimeError(f"restart timing segments are inconsistent: {path}")

    values = {key: _nonnegative_integer(properties, key) for key in INTEGER_PROPERTIES}
    callbacks = {
        "discover": values["restart_discover_callback_count"],
        "allocate": values["restart_allocate_callback_count"],
    }
    if callbacks != {"discover": 2, "allocate": 2}:
        raise RuntimeError(f"restart callback cardinalities are inconsistent: {path}")
    if not (
        values["restart_initial_run_generation"] > 0
        and values["restart_initial_run_generation"]
        == values["restart_successor_run_generation"]
        == values["restart_published_run_generation"]
        and values["restart_initial_rate_generation"] > 0
        and values["restart_successor_rate_generation"] > 0
        and values["restart_initial_rate_generation"] != values["restart_successor_rate_generation"]
        and values["restart_successor_rate_generation"]
        == values["restart_published_rate_generation"]
    ):
        raise RuntimeError(f"restart generation facts are inconsistent: {path}")

    physical = values["restart_physical_slot_count"]
    active = values["restart_active_entry_limit"]
    span = values["restart_logical_requested_payload_byte_span_per_slot"]
    logical_bytes = physical * span
    if not (
        properties.get("restart_allocation_action") == "reused_larger"
        and values["restart_retained_allocation_generation"] > 0
        and values["restart_retained_allocation_generation"]
        == values["restart_previous_allocation_generation"]
        == values["restart_maximum_allocation_generation"]
        and values["restart_storage_binding_generation"] >= 2
        and values["restart_materialization_count"] == 1
        and values["restart_live_materialization_count"] == 1
        and values["restart_peak_live_materialization_count"] == 1
        and 0 < active < physical
        and values["restart_live_physical_entry_count"] == physical
        and values["restart_peak_live_physical_entry_count"] == physical
        and span > 0
        and values["restart_live_logical_requested_payload_bytes"] == logical_bytes
        and values["restart_peak_live_logical_requested_payload_bytes"] == logical_bytes
        and values["restart_provider_materialized_entries"] == physical
        and values["restart_post_stop_live_materialization_count"] == 0
        and values["restart_post_stop_live_physical_entry_count"] == 0
        and values["restart_post_stop_live_logical_requested_payload_bytes"] == 0
        and 0
        < values["restart_minimum_pool_generation"]
        < values["restart_maximum_pool_generation"]
        == values["restart_maximum_published_pool_generation"]
    ):
        raise RuntimeError(f"restart resource invariants are inconsistent: {path}")

    reservation_reason = properties.get("restart_provider_reserved_bytes_unavailable_reason")
    if (
        properties.get("restart_provider_reserved_bytes_available") != "false"
        or not isinstance(reservation_reason, str)
        or not reservation_reason
        or "restart_provider_reserved_bytes_total" in properties
    ):
        raise RuntimeError(f"provider reservation availability is misrepresented: {path}")

    return {
        "timings_ns": timings,
        "callbacks": callbacks,
        "resources": {
            "allocation_action": "reused_larger",
            "retained_allocation_generation": values["restart_retained_allocation_generation"],
            "materialization_count": values["restart_materialization_count"],
            "live_materialization_count": values["restart_live_materialization_count"],
            "peak_live_materialization_count": values["restart_peak_live_materialization_count"],
            "physical_slot_count": physical,
            "provider_materialized_entries": values["restart_provider_materialized_entries"],
            "active_entry_limit": active,
            "live_physical_entry_count": values["restart_live_physical_entry_count"],
            "peak_live_physical_entry_count": values["restart_peak_live_physical_entry_count"],
            "logical_requested_payload_bytes": {
                "live": values["restart_live_logical_requested_payload_bytes"],
                "peak": values["restart_peak_live_logical_requested_payload_bytes"],
                "byte_span_per_slot": span,
                "scope": "requested_payload_only_excludes_provider_overhead",
            },
            "provider_reserved_bytes": {
                "available": False,
                "bytes": None,
                "reason": reservation_reason,
            },
            "post_stop_live": {
                "materializations": values["restart_post_stop_live_materialization_count"],
                "physical_entries": values["restart_post_stop_live_physical_entry_count"],
                "logical_requested_payload_bytes": values[
                    "restart_post_stop_live_logical_requested_payload_bytes"
                ],
            },
        },
        "generations": {
            key.removeprefix("restart_"): values[key]
            for key in INTEGER_PROPERTIES
            if "generation" in key
        },
    }


def run_case(
    binary: Path,
    frozen_sha256: str,
    scenario: ScenarioProbe,
    repetition: int,
    output_dir: Path,
    timeout: float,
) -> dict[str, object]:
    case_dir = output_dir / f"repetition_{repetition:03d}" / scenario.slug
    case_dir.mkdir(parents=True, exist_ok=True)
    gtest_json = case_dir / "gtest.json"
    gtest_json.unlink(missing_ok=True)
    command = [
        str(binary),
        f"--gtest_filter={scenario.gtest_filter}",
        "--gtest_color=no",
        f"--gtest_output=json:{gtest_json}",
    ]
    require_unchanged_binary(binary, frozen_sha256, "runtime_test")
    try:
        completed = evidence.run_logged_child(command, case_dir, timeout)
    finally:
        require_unchanged_binary(binary, frozen_sha256, "runtime_test")
    if completed.returncode != 0:
        raise RuntimeError(f"restart conformance probe exited {completed.returncode}: {case_dir}")
    if not gtest_json.is_file():
        raise RuntimeError(f"restart conformance probe produced no fresh gtest JSON: {case_dir}")
    result = (
        checked_gtest_result(gtest_json)
        if scenario.instrumented
        else checked_gtest_pass(gtest_json, scenario.gtest_filter)
    )
    result["scenario"] = scenario.slug
    result["repetition"] = repetition
    result["raw_gtest_json"] = str(gtest_json.relative_to(output_dir))
    result["raw_gtest_json_sha256"] = file_sha256(gtest_json)
    return result


def run_hsb_case(
    binary: Path,
    frozen_sha256: str,
    scenario: HsbScenarioProbe,
    repetition: int,
    output_dir: Path,
    timeout: float,
) -> dict[str, object]:
    case_dir = output_dir / "hsb_shaped" / f"repetition_{repetition:03d}" / scenario.slug
    case_dir.mkdir(parents=True, exist_ok=True)
    command = [str(binary), scenario.argument]
    require_unchanged_binary(binary, frozen_sha256, "HSB validation")
    try:
        completed = evidence.run_logged_child(command, case_dir, timeout)
    finally:
        require_unchanged_binary(binary, frozen_sha256, "HSB validation")
    stdout = case_dir / "stdout.log"
    stderr = case_dir / "stderr.log"
    command_path = case_dir / "command.json"
    if completed.returncode != 0:
        raise RuntimeError(
            f"HSB-shaped boundary assertion exited {completed.returncode}: {case_dir}"
        )
    return {
        "scenario": scenario.slug,
        "repetition": repetition,
        "assertion_outcome": "passed",
        "command": str(command_path.relative_to(output_dir)),
        "stdout": str(stdout.relative_to(output_dir)),
        "stdout_sha256": file_sha256(stdout),
        "stderr": str(stderr.relative_to(output_dir)),
        "stderr_sha256": file_sha256(stderr),
    }


def make_hsb_boundary(
    args: argparse.Namespace,
    binary: Path | None,
    runs: list[dict[str, object]] | None,
    binary_sha256: str | None,
) -> dict[str, object]:
    remaining_gap = (
        "actual_hsb_camera_provider_matrix_with_timing_and_allocator_registration_counters"
    )
    if binary is None or runs is None:
        return {
            "status": "not_collected",
            "shape": "generated_heterogeneous_three_field_camera_like_bundle",
            "boundary": "process_local_generated_schema_not_hsb_transport",
            "reason": "--hsb-validation-binary was not supplied",
            "remaining_gap": remaining_gap,
            "scenarios": [],
        }
    if binary_sha256 is None:
        raise RuntimeError("HSB-shaped boundary binary hash is missing")

    rows = []
    for scenario in HSB_SCENARIO_PROBES:
        scenario_runs = [run for run in runs if run["scenario"] == scenario.slug]
        if len(scenario_runs) != args.repetitions:
            raise RuntimeError(f"HSB-shaped boundary evidence is incomplete for {scenario.slug}")
        rows.append(
            {
                "scenario": scenario.slug,
                "argument": scenario.argument,
                "assertion_contract": scenario.assertion_contract,
                "passing_repetitions": len(scenario_runs),
                "measurement_scope": "focused_standalone_assertions",
                "timings_ns": None,
                "timings_unavailable_reason": (
                    "standalone_validation_scenario_does_not_export_timing_properties"
                ),
                "allocator_call_count": None,
                "allocator_call_count_unavailable_reason": (
                    "standalone_validation_scenario_asserts_materialization_identity_but_does_not_export_calls"
                ),
                "registration_call_count": None,
                "registration_call_count_unavailable_reason": (
                    "direct_host_generated_field_provider_has_no_exported_registration_counter"
                ),
                "raw_runs": scenario_runs,
            }
        )
    return {
        "status": "assertion_backed_partial_coverage",
        "shape": "generated_heterogeneous_three_field_camera_like_bundle",
        "boundary": "process_local_generated_schema_not_hsb_transport",
        "binary": {
            "path": str(binary),
            "sha256": binary_sha256,
            "build": build_metadata(binary),
        },
        "remaining_gap": remaining_gap,
        "scenarios": rows,
    }


def make_summary(
    args: argparse.Namespace,
    binary: Path,
    runs: list[dict[str, object]],
    binary_sha256: str,
    hsb_binary: Path | None = None,
    hsb_runs: list[dict[str, object]] | None = None,
    hsb_binary_sha256: str | None = None,
) -> dict:
    instrumented_runs = [run for run in runs if run["scenario"] == "smaller_compatible"]
    if len(instrumented_runs) != args.repetitions:
        raise RuntimeError("restart matrix lacks the complete instrumented smaller-rate scenario")
    timing_summary = {
        name: summarize([int(run["timings_ns"][name]) for run in instrumented_runs])
        for name in TIMING_PROPERTIES
    }
    matrix = []
    for scenario in SCENARIO_PROBES:
        scenario_runs = [run for run in runs if run["scenario"] == scenario.slug]
        if len(scenario_runs) != args.repetitions:
            raise RuntimeError(f"restart matrix is incomplete for scenario {scenario.slug}")
        row: dict[str, object] = {
            "scenario": scenario.slug,
            "gtest_filter": scenario.gtest_filter,
            "assertion_contract": scenario.assertion_contract,
            "passing_repetitions": len(scenario_runs),
            "measurement_scope": (
                "instrumented_properties" if scenario.instrumented else "focused_gtest_assertions"
            ),
            "evidence_boundary": scenario.evidence_boundary,
        }
        if scenario.instrumented:
            row["timings_ns"] = timing_summary
            row["timings_unavailable_reason"] = None
        else:
            row["timings_ns"] = None
            row["timings_unavailable_reason"] = "focused_gtest_does_not_record_timing_properties"
        matrix.append(row)
    return {
        "schema": SCHEMA,
        "qualification": {
            "status": "instrumented_conformance_probe_characterization_only",
            "product_latency_threshold": None,
            "notes": [
                "each scenario repetition is one isolated process running its exact "
                "conformance test",
                "five required outcome rows plus one late-stage assertion compose exact focused "
                "gtests from one binary "
                "and source identity",
                "the incompatible replacement row uses a synthetic compatibility fault, "
                "not an actual layout",
                "timings characterize the instrumented test path and are not a product SLO",
                "non-instrumented matrix rows report unavailable timing instead of inferring it",
                "provider reservation bytes are unavailable and are not inferred",
            ],
        },
        "probe": {
            "gtest_filters": [scenario.gtest_filter for scenario in SCENARIO_PROBES],
            "repetitions": args.repetitions,
            "process_timeout_sec": args.process_timeout_sec,
        },
        "source": {
            "git_revision": args.git_revision,
            "source_state": args.source_state,
            "source_snapshot_sha256": args.source_snapshot_sha256,
        },
        "binary": {
            "path": str(binary),
            "sha256": binary_sha256,
            "build": build_metadata(binary),
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
        },
        "timings_ns": timing_summary,
        "restart_matrix": matrix,
        "hsb_shaped_boundary": make_hsb_boundary(args, hsb_binary, hsb_runs, hsb_binary_sha256),
        "callback_observations": [run["callbacks"] for run in instrumented_runs],
        "resource_observations": [run["resources"] for run in instrumented_runs],
        "generation_observations": [run["generations"] for run in instrumented_runs],
        "raw_runs": [
            {
                "scenario": run["scenario"],
                "repetition": run["repetition"],
                "raw_gtest_json": run["raw_gtest_json"],
                "raw_gtest_json_sha256": run["raw_gtest_json_sha256"],
                "timings_ns": run.get("timings_ns"),
                "callbacks": run.get("callbacks"),
                "assertion_outcome": run.get("assertion_outcome", "passed"),
            }
            for run in runs
        ],
    }


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        binary = args.runtime_test_binary.resolve(strict=True)
        hsb_binary = (
            args.hsb_validation_binary.resolve(strict=True)
            if args.hsb_validation_binary is not None
            else None
        )
        binary_sha256 = file_sha256(binary)
        hsb_binary_sha256 = file_sha256(hsb_binary) if hsb_binary is not None else None
        if args.output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="holoscan-rate-restart-probe-"))
        else:
            output_dir = args.output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        runs = [
            run_case(
                binary,
                binary_sha256,
                scenario,
                repetition,
                output_dir,
                args.process_timeout_sec,
            )
            for repetition in range(1, args.repetitions + 1)
            for scenario in SCENARIO_PROBES
        ]
        hsb_runs = (
            [
                run_hsb_case(
                    hsb_binary,
                    hsb_binary_sha256,
                    scenario,
                    repetition,
                    output_dir,
                    args.process_timeout_sec,
                )
                for repetition in range(1, args.repetitions + 1)
                for scenario in HSB_SCENARIO_PROBES
            ]
            if hsb_binary is not None
            else None
        )
        require_unchanged_binary(binary, binary_sha256, "runtime_test")
        if hsb_binary is not None and hsb_binary_sha256 is not None:
            require_unchanged_binary(hsb_binary, hsb_binary_sha256, "HSB validation")
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                make_summary(
                    args,
                    binary,
                    runs,
                    binary_sha256,
                    hsb_binary,
                    hsb_runs,
                    hsb_binary_sha256,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(summary_path)
        return 0
    except (
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
