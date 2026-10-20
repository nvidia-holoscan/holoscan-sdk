# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(BENCHMARK_DIR / "runtime"))


def load_module(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, BENCHMARK_DIR / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module("transport_suite_runner", "run_transport_benchmark_suite.py")
summarizer = load_module("transport_suite_summarizer", "summarize_transport_benchmark_suite.py")
docs_summarizer = load_module("runtime_docs_summarizer", "summarize_runtime_benchmarks.py")
graph_runner = load_module("docs_preflight_graph_runner", "runtime/run_runtime_graph_benchmark.py")
graph_interprocess_runner = load_module(
    "docs_preflight_graph_interprocess_runner",
    "runtime/run_runtime_graph_interprocess_benchmark.py",
)


def suite_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        build_dir=tmp_path / "build",
        output_dir=tmp_path / "results",
        preset="smoke",
        groups=list(runner.GROUPS),
        cpus=list(range(8)),
        latency_period_us=200,
        tensor_bytes=1 << 20,
        repetitions=1,
        skip_gpu=False,
    )


def docs_contract() -> tuple[dict[str, object], list[dict[str, object]]]:
    case_ids = sorted(docs_summarizer.expected_case_ids())
    manifest: dict[str, object] = {
        "preset": docs_summarizer.DOCS_PRESET,
        "repetitions": 7,
        "cases": [{"case_id": case_id, "status": "passed"} for case_id in case_ids],
    }
    rows: list[dict[str, object]] = [
        {"native_runs": [{"manifest_case_id": case_id}]} for case_id in case_ids
    ]
    return manifest, rows


def test_graph_intraprocess_image_command_includes_tensor_options(tmp_path: Path) -> None:
    # Key setup: build one intraprocess Image command through the shared Graph command helper.
    args = suite_args(tmp_path)
    command = runner.graph_intraprocess_command(
        args,
        args.build_dir / "benchmark/runtime/runtime_graph_benchmark",
        args.output_dir / "graph/intraprocess/image",
        "latency",
        "image",
        storage="device",
        bulk_work_profile="descriptor_only",
    )

    # Key assertion: Image uses the descriptor-backed Tensor transport options unchanged.
    assert command[command.index("--tensor-storage") + 1] == "device"
    assert command[command.index("--tensor-bytes") + 1] == str(args.tensor_bytes)
    assert command[command.index("--bulk-work-profile") + 1] == "descriptor_only"


def test_current_suite_plan_covers_every_maintained_transport_group(tmp_path: Path) -> None:
    # Key setup: construct the complete smoke plan without requiring built binaries.
    cases = runner.build_cases(suite_args(tmp_path))
    by_id = {case.case_id: case for case in cases}

    # Key assertion: the plan spans all five existing result producers and known topology limits.
    assert len(cases) == 87
    assert {case.layer for case in cases} == {"holomq", "channel", "graph"}
    assert "holomq_interprocess_inline_64_throughput_q1" in by_id
    assert "holomq_interprocess_inline_64_throughput_q64" in by_id
    assert "holomq_interprocess_inline_64_1to2_throughput" in by_id
    assert "holomq_interprocess_buffer_pageable_host_latency" in by_id
    assert "holomq_interprocess_buffer_pinned_host_latency" in by_id
    assert "holomq_intraprocess_concurrent_1to1" in by_id
    assert "channel_matrix_latency_rep000" in by_id
    assert "graph_intraprocess_tensor_pageable_host_latency" in by_id
    assert "graph_intraprocess_pod_throughput_q1" in by_id
    assert "graph_intraprocess_pod_throughput_q64" in by_id
    assert "graph_intraprocess_chain_8_throughput" in by_id
    assert "graph_interprocess_1to2_latency" in by_id
    assert "graph_interprocess_1to1_throughput_q8" in by_id
    assert "graph_interprocess_schema_1to1_latency" in by_id
    assert "graph_interprocess_tensor_pinned_host_1to1_throughput" in by_id
    assert "graph_interprocess_image_device_1to1_latency" in by_id
    assert "graph_interprocess_1to2_throughput" in by_id
    assert "graph_interprocess_tensor_device_1to2_latency" not in by_id
    assert "graph_interprocess_image_device_chain_latency" not in by_id
    assert "graph_intraprocess_pod_busy_poll_latency" in by_id
    assert "graph_interprocess_1to1_busy_poll_latency" in by_id
    for relays in (1, 2, 4, 8):
        for workload in ("throughput", "latency"):
            command = by_id[f"graph_intraprocess_chain_{relays}_{workload}"].command
            assert command[command.index("--relay-count") + 1] == str(relays)
            assert "--partition-algorithm" not in command
    assert "--consumer-cpus" in by_id["graph_interprocess_1to2_latency"].command
    fanout_command = by_id["graph_interprocess_1to2_latency"].command
    assert fanout_command[fanout_command.index("--publisher-period-us") + 1] == "5000"
    assert fanout_command[fanout_command.index("--queue-depth") + 1] == "2"
    one_to_one_command = by_id["graph_interprocess_1to1_latency"].command
    assert one_to_one_command[one_to_one_command.index("--queue-depth") + 1] == "1"
    saturated_fanout_command = by_id["graph_interprocess_1to2_throughput"].command
    assert (
        saturated_fanout_command[saturated_fanout_command.index("--publisher-period-us") + 1] == "0"
    )
    assert saturated_fanout_command[saturated_fanout_command.index("--queue-depth") + 1] == "2"
    pageable_command = by_id["holomq_interprocess_buffer_pageable_host_latency"].command
    assert pageable_command[pageable_command.index("--storage") + 1] == "pageable_host"
    pinned_command = by_id["holomq_interprocess_buffer_pinned_host_latency"].command
    assert pinned_command[pinned_command.index("--storage") + 1] == "pinned_host"
    device_command = by_id["holomq_interprocess_buffer_device_latency"].command
    assert device_command[device_command.index("--publisher-period-us") + 1] == "500"
    assert device_command[device_command.index("--device-sync") + 1] == "ready"
    assert "holomq_interprocess_buffer_device_ready_latency" not in by_id
    concurrent_command = by_id["holomq_intraprocess_concurrent_1to1"].command
    assert "HOLOMQ_BENCHMARK_PUBLISHER_CPU=0" in concurrent_command
    assert "HOLOMQ_BENCHMARK_SUBSCRIBER_CPU=1" in concurrent_command
    assert "--busy-poll" in by_id["graph_intraprocess_pod_busy_poll_latency"].command
    assert "--busy-poll" in by_id["graph_interprocess_1to1_busy_poll_latency"].command
    for case_id in (
        "graph_intraprocess_pod_1to2_latency",
        "graph_intraprocess_schema_1to2_latency",
    ):
        command = by_id[case_id].command
        assert command[command.index("--queue-depth") + 1] == "2"
    schema_command = by_id["graph_interprocess_schema_1to1_latency"].command
    assert schema_command[schema_command.index("--payload") + 1] == "schema"
    tensor_command = by_id["graph_interprocess_tensor_pinned_host_1to1_throughput"].command
    assert tensor_command[tensor_command.index("--tensor-storage") + 1] == "pinned_host"
    assert tensor_command[tensor_command.index("--bulk-work-profile") + 1] == "descriptor_only"
    # Each pinned-host Tensor and Image row names its writer, with a CUDA-written twin.
    for payload in ("tensor", "image"):
        for case_suffix, writer in (("", "host"), ("_cuda_writer", "cuda")):
            command = by_id[
                f"graph_interprocess_{payload}_pinned_host{case_suffix}_1to1_throughput"
            ].command
            assert command[command.index("--pinned-host-writer") + 1] == writer
    image_command = by_id["graph_interprocess_image_device_1to1_latency"].command
    assert "--pinned-host-writer" not in image_command
    assert image_command[image_command.index("--payload") + 1] == "image"
    assert image_command[image_command.index("--tensor-storage") + 1] == "device"
    assert image_command[image_command.index("--bulk-work-profile") + 1] == "full_write"
    validation_commands = [case.command for case in cases if "--payload-validation" in case.command]
    assert validation_commands
    assert all(
        command[command.index("--payload-validation") + 1] in {"all", "final"}
        for command in validation_commands
    )
    assert runner.WORKLOAD_GEOMETRY_POLICY["throughput_queue_depths"] == [1, 2, 8, 64]


def test_production_plan_disables_payload_scans_and_records_sampling_policy(
    tmp_path: Path,
) -> None:
    # Key setup: build the production plan for every sample-driven benchmark layer.
    args = suite_args(tmp_path)
    args.preset = "production"
    args.repetitions = 7
    cases = runner.build_cases(args)
    by_id = {case.case_id: case for case in cases}

    # Key assertion: production commands suppress content scans while retaining structural checks.
    validation_commands = [case.command for case in cases if "--payload-validation" in case.command]
    assert validation_commands
    assert all(
        command[command.index("--payload-validation") + 1] == "none"
        for command in validation_commands
    )
    assert runner.PRODUCTION_MEASUREMENT_POLICY["repetitions"] == 7
    assert runner.PRODUCTION_MEASUREMENT_POLICY["inline"]["latency_samples"] >= 10_000
    assert runner.PRODUCTION_MEASUREMENT_POLICY["percentiles"]["p999_minimum_samples"] == 10_000
    intraprocess_holomq = by_id["graph_intraprocess_pod_holomq_throughput"].command
    assert intraprocess_holomq[intraprocess_holomq.index("--session-timeout-ms") + 1] == "120000"
    assert intraprocess_holomq[intraprocess_holomq.index("--process-timeout-sec") + 1] == "150"
    tensor_latency = by_id["graph_interprocess_tensor_device_1to1_latency"].command
    assert tensor_latency[tensor_latency.index("--bulk-work-profile") + 1] == "descriptor_only"
    intraprocess_tensor_latency = by_id["graph_intraprocess_tensor_device_latency"].command
    assert (
        intraprocess_tensor_latency[intraprocess_tensor_latency.index("--bulk-work-profile") + 1]
        == "descriptor_only"
    )
    holomq_device = by_id["holomq_interprocess_buffer_device_latency"].command
    assert holomq_device[holomq_device.index("--device-sync") + 1] == "none"
    assert holomq_device[holomq_device.index("--publisher-period-us") + 1] == "200"
    holomq_device_ready = by_id["holomq_interprocess_buffer_device_ready_latency"].command
    assert holomq_device_ready[holomq_device_ready.index("--device-sync") + 1] == "ready"
    assert holomq_device_ready[holomq_device_ready.index("--publisher-period-us") + 1] == "500"


def test_docs_core_runtime_plan_contains_only_documented_production_rows(
    tmp_path: Path,
) -> None:
    # Key setup: construct the dedicated docs plan with the requested eight physical CPUs.
    args = suite_args(tmp_path)
    args.preset = runner.DOCS_CORE_RUNTIME_PRESET
    args.repetitions = 7
    cases = runner.build_cases(args)
    by_id = {case.case_id: case for case in cases}

    # Key assertion: the exact inventory uses the documented counts and both fanout placements.
    assert len(cases) == 48
    assert set(by_id) == docs_summarizer.expected_case_ids()
    assert {case.layer for case in cases} == {"graph"}
    assert {case.domain for case in cases} == {"intraprocess", "interprocess"}
    assert all("--repetitions" in case.command for case in cases)
    assert all(case.command[case.command.index("--repetitions") + 1] == "7" for case in cases)
    for fanout in (2, 4):
        for workload, required_cpus in (("latency", 1), ("throughput", 2)):
            shared = by_id[f"graph_intraprocess_docs_pod_1to{fanout}_{workload}"]
            command = shared.command
            assert command[command.index("--operator-lanes") + 1] == "shared"
            assert "--sink-cpus" not in command
            assert "--source-cpu" not in command
            assert command[command.index("--controller-cpu") + 1] == "0"
            assert "--channel-pump-cpu" not in command
            assert "--force-holomq-event-wake" not in command
            assert shared.required_cpus == required_cpus
            if workload == "throughput":
                assert command[command.index("--notification-driver-cpu") + 1] == "1"
            else:
                assert "--notification-driver-cpu" not in command
    separate_fanout = by_id["graph_intraprocess_docs_pod_1to4_per_sink_throughput"].command
    assert separate_fanout[separate_fanout.index("--operator-lanes") + 1] == "per_sink"
    assert separate_fanout[separate_fanout.index("--sink-cpus") + 1] == "1,2,3,4"
    assert separate_fanout[separate_fanout.index("--controller-cpu") + 1] == "5"
    # compile() delivers this POD fanout inside emit(), so its reserved pump CPU stays unused.
    assert "--channel-pump-cpu" not in separate_fanout
    assert separate_fanout[separate_fanout.index("--notification-driver-cpu") + 1] == "7"
    chain = by_id["graph_intraprocess_docs_chain_10_operators_latency"].command
    assert chain[chain.index("--relay-count") + 1] == "8"
    intraprocess_multi_tensor = by_id[
        "graph_intraprocess_docs_multi_tensor_pageable_host_1to1_latency"
    ].command
    assert intraprocess_multi_tensor[intraprocess_multi_tensor.index("--scenario") + 1] == (
        "multi_tensor"
    )
    assert (
        intraprocess_multi_tensor[intraprocess_multi_tensor.index("--bulk-work-profile") + 1]
        == "descriptor_only"
    )
    for case_id in (
        "graph_intraprocess_docs_tensor_pinned_host_1to1_latency",
        "graph_intraprocess_docs_tensor_pinned_host_1to1_throughput",
        "graph_intraprocess_docs_tensor_device_1to1_latency",
        "graph_intraprocess_docs_tensor_device_1to1_throughput",
        "graph_intraprocess_docs_image_pageable_host_1to1_latency",
        "graph_intraprocess_docs_image_pageable_host_1to1_throughput",
        "graph_intraprocess_docs_image_pageable_host_1to1_borrowed_latency",
        "graph_intraprocess_docs_image_pageable_host_1to1_borrowed_throughput",
        "graph_intraprocess_docs_multi_tensor_pageable_host_1to1_latency",
        "graph_intraprocess_docs_multi_tensor_pageable_host_1to1_throughput",
    ):
        command = by_id[case_id].command
        assert command[command.index("--bulk-work-profile") + 1] == "descriptor_only"
    for workload in ("latency", "throughput"):
        owning = by_id[f"graph_intraprocess_docs_image_pageable_host_1to1_{workload}"].command
        borrowed = by_id[
            f"graph_intraprocess_docs_image_pageable_host_1to1_borrowed_{workload}"
        ].command
        assert "--borrow-image-input" not in owning
        assert "--borrow-image-input" in borrowed
        assert borrowed[borrowed.index("--scenario") + 1] == "image"
    multi_tensor = by_id[
        "graph_interprocess_docs_multi_tensor_loan_pageable_host_1to1_latency"
    ].command
    assert multi_tensor[multi_tensor.index("--payload") + 1] == "multi_tensor_loan"
    assert multi_tensor[multi_tensor.index("--bulk-work-profile") + 1] == "descriptor_only"
    assert multi_tensor[multi_tensor.index("--samples") + 1] == "1000"
    multi_tensor_throughput = by_id[
        "graph_interprocess_docs_multi_tensor_loan_pageable_host_1to1_throughput"
    ].command
    assert multi_tensor_throughput[multi_tensor_throughput.index("--samples") + 1] == "20000"
    pageable_image = by_id["graph_interprocess_docs_image_pageable_host_1to1_latency"].command
    assert pageable_image[pageable_image.index("--payload") + 1] == "image"
    assert pageable_image[pageable_image.index("--tensor-storage") + 1] == "pageable_host"
    assert pageable_image[pageable_image.index("--bulk-work-profile") + 1] == "descriptor_only"
    interprocess_latency = by_id["graph_interprocess_docs_pod_1to2_latency"].command
    assert interprocess_latency[interprocess_latency.index("--samples") + 1] == "1000"
    assert interprocess_latency[interprocess_latency.index("--warmup") + 1] == "1000"
    intraprocess_latency = by_id["graph_intraprocess_docs_pod_1to4_latency"].command
    assert intraprocess_latency[intraprocess_latency.index("--samples") + 1] == "10000"


@pytest.mark.parametrize(("cpu_count", "planned_count"), [(8, 48), (4, 19)])
def test_docs_core_runtime_plan_passes_child_runner_preflight(
    tmp_path: Path, cpu_count: int, planned_count: int
) -> None:
    # Key setup: generate full and compact docs commands and parse them with their child runners.
    args = suite_args(tmp_path)
    args.preset = runner.DOCS_CORE_RUNTIME_PRESET
    args.repetitions = 7
    args.cpus = list(range(cpu_count))
    cases = runner.build_cases(args)
    planned = [case for case in cases if case.status == "planned"]
    validators = {
        runner.GRAPH_RUNNER.resolve(): (graph_runner.parse_args, graph_runner.validate_args),
        runner.GRAPH_INTERPROCESS_RUNNER.resolve(): (
            graph_interprocess_runner.parse_args,
            graph_interprocess_runner.validate_args,
        ),
    }

    # Key assertion: parser and semantic validation accept every runnable child command.
    assert len(cases) == 48
    assert len(planned) == planned_count
    for case in planned:
        child = Path(case.command[1]).resolve()
        assert child in validators, case.case_id
        parse_child_args, validate_child_args = validators[child]
        try:
            child_args = parse_child_args(case.command[2:])
            validate_child_args(child_args)
        except (SystemExit, ValueError) as error:
            pytest.fail(f"child-command preflight failed for {case.case_id}: {error}")


@pytest.mark.parametrize(("cpu_count", "planned_count"), [(4, 19), (6, 42)])
def test_docs_core_runtime_plan_omits_only_unsupported_cpu_topologies(
    tmp_path: Path, cpu_count: int, planned_count: int
) -> None:
    # Key setup: plan the docs suite with a reduced set of distinct physical cores.
    args = suite_args(tmp_path)
    args.preset = runner.DOCS_CORE_RUNTIME_PRESET
    args.repetitions = 7
    args.cpus = list(range(cpu_count))
    cases = runner.build_cases(args)
    by_id = {case.case_id: case for case in cases}
    planned = [case for case in cases if case.status == "planned"]
    skipped = [case for case in cases if case.status == "skipped"]

    # Key assertion: all 48 identities remain auditable while only fitting rows get commands.
    assert len(cases) == 48
    assert len(planned) == planned_count
    assert len(skipped) == 48 - planned_count
    assert all(case.command for case in planned)
    assert all(
        not case.command
        and case.returncode == 77
        and case.required_cpus is not None
        and case.required_cpus > cpu_count
        and case.skip_reason
        for case in skipped
    )
    if cpu_count == 4:
        direct = by_id["graph_intraprocess_docs_pod_1to1_throughput"].command
        assert direct[direct.index("--controller-cpu") + 1] == "2"
        assert direct[direct.index("--notification-driver-cpu") + 1] == "3"
        assert by_id["graph_intraprocess_docs_pod_1to4_throughput"].status == "planned"
        assert by_id["graph_intraprocess_docs_chain_10_operators_throughput"].status == "skipped"
        assert all(case.status == "skipped" for case in cases if case.domain == "interprocess")
    else:
        assert by_id["graph_intraprocess_docs_pod_1to2_per_sink_throughput"].status == "planned"
        assert by_id["graph_intraprocess_docs_pod_1to4_per_sink_latency"].status == "skipped"
        assert by_id["graph_interprocess_docs_pod_1to1_throughput"].status == "planned"
        assert by_id["graph_interprocess_docs_pod_1to2_latency"].status == "skipped"
        assert by_id["graph_interprocess_docs_chain_3_operators_throughput"].status == "skipped"


def test_runtime_docs_summarizer_renders_requested_tables_and_percentiles() -> None:
    # Key setup: provide one normalized production row for every docs case identifier.
    rows = {
        case_id: {
            "application_payload_bytes": (2 << 20 if "multi_tensor" in case_id else (1 << 20)),
            "latency_p50_ns": 1_234.0,
            "latency_p99_ns": 2_345.0,
            "latency_p999_ns": 3_456.0,
            "logical_messages_per_second": 12_345.0,
            "aggregate_deliveries_per_second": 49_380.0,
            "primary_metric_coefficient_of_variation": 0.01,
            "event_wake": True,
            "runtime_support_threads": "controller,channel_pump",
            "delivery_path": "holomq_inline",
        }
        for case_id in docs_summarizer.expected_case_ids()
    }
    rows["graph_intraprocess_docs_pod_1to2_throughput"][
        "primary_metric_coefficient_of_variation"
    ] = 0.228
    for fanout in (2, 4):
        for workload in ("latency", "throughput"):
            shared = rows[f"graph_intraprocess_docs_pod_1to{fanout}_{workload}"]
            shared["event_wake"] = False
            shared["delivery_path"] = "direct_retained"
            shared["runtime_support_threads"] = (
                "controller" if workload == "latency" else "controller,notification_driver"
            )

    latency, throughput = docs_summarizer.render_tables(rows)

    # Key assertion: only the sufficiently sampled domain reports p99.9.
    single_process_latency, interprocess_latency = latency.split(
        "### Interprocess Graph transport latency"
    )
    assert "| p50 | p99 | p99.9 |" in single_process_latency
    assert "| p50 | p99 | Notes |" in interprocess_latency
    assert "p99.9" not in interprocess_latency
    assert "`schema::ImageT` | Pageable host" in interprocess_latency
    assert "Single-process 1:4" in latency
    assert "Shared operator partition; direct-retained fanout; eventless; pump-free" in latency
    assert (
        "One source partition and one partition per subscriber; HoloMQ fanout; event wake; "
        "channel pump" in latency
    )
    assert "Single-process 10-operator chain" in latency
    assert "runtime_benchmark::MultiTensorMessageT" in latency
    assert "3.456 µs" in latency
    assert "Single-process Graph throughput" in throughput
    assert "Interprocess Graph throughput" in throughput
    assert "49,380" in throughput
    assert "run-to-run CV 22.8% exceeds the 10% publication gate" in throughput


def test_runtime_docs_summarizer_accepts_audited_reduced_core_omissions() -> None:
    # Key setup: omit one eight-core row from a six-core run with explicit manifest evidence.
    manifest, rows = docs_contract()
    manifest["cpus"] = list(range(6))
    omitted_id = "graph_intraprocess_docs_pod_1to4_per_sink_throughput"
    cases = manifest["cases"]
    assert isinstance(cases, list)
    omitted_case = next(case for case in cases if case["case_id"] == omitted_id)
    omitted_case.update(
        status="skipped",
        returncode=77,
        command=[],
        required_cpus=8,
        skip_reason="topology requires 8 distinct physical CPUs; only 6 selected",
    )
    rows = [row for row in rows if row["native_runs"][0]["manifest_case_id"] != omitted_id]

    # Key assertion: the skipped identity remains in the contract but needs no normalized row.
    by_id = docs_summarizer.require_complete_docs_run(manifest, rows)
    assert len(by_id) == 47
    assert omitted_id not in by_id


@pytest.mark.parametrize("duplicate_source", ["manifest", "rows"])
def test_runtime_docs_summarizer_rejects_duplicate_case_ids(duplicate_source: str) -> None:
    # Key setup: duplicate one otherwise complete case in either contract input.
    manifest, rows = docs_contract()
    if duplicate_source == "manifest":
        cases = manifest["cases"]
        assert isinstance(cases, list)
        cases.append(dict(cases[0]))
    else:
        rows.append(dict(rows[0]))

    # Key assertion: duplicate evidence is rejected before set/dict collapsing can mask it.
    with pytest.raises(RuntimeError, match="duplicate case identifiers"):
        docs_summarizer.require_complete_docs_run(manifest, rows)


def test_runtime_docs_update_renders_provenance_from_suite_artifacts(tmp_path: Path) -> None:
    # Key setup: provide one complete docs contract and the native manifest for its widest
    # CPU-placement row.
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    (suite_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    manifest, rows = docs_contract()
    manifest.update(
        {
            "created_unix_seconds": 0,
            "cpus": list(range(8)),
        }
    )
    for row in rows:
        native_runs = row["native_runs"]
        assert isinstance(native_runs, list)
        case_id = native_runs[0]["manifest_case_id"]
        assert isinstance(case_id, str)
        row.update(
            {
                "primary_metric_coefficient_of_variation": 0.01,
                "payload_validation": "none",
                "measured_samples": (
                    1_000 if "interprocess" in case_id and case_id.endswith("_latency") else 10_000
                ),
                "warmup_samples": 1_000,
            }
        )
        if case_id == docs_summarizer.PROVENANCE_CASE_ID:
            native_runs[0]["artifact"] = "graph/widest/summary.json"
    native_dir = suite_dir / "graph/widest"
    native_dir.mkdir(parents=True)
    write_json(
        native_dir / "manifest.json",
        {
            "machine": "x86_64",
            "kernel": "6.8.0-test",
            "cpu_topology": [
                {
                    "cpu": cpu,
                    "pinned": True,
                    "package_id": 0,
                    "core_id": cpu,
                    "thread_siblings": f"{cpu},{cpu + 8}",
                    "governor": "performance",
                }
                for cpu in range(8)
            ],
            "gpus": [{"name": "Test GPU", "driver_version": "1.2.3"}],
            "build": {
                "cmake_build_type": "Release",
                "cxx_compiler_version": "Test C++ 1.0",
                "cuda_compiler_version_output": "Test CUDA 2.0",
            },
        },
    )
    by_id = docs_summarizer.require_complete_docs_run(manifest, rows)
    provenance = docs_summarizer.render_provenance(suite_dir, manifest, by_id)
    document = "\n".join(
        (
            docs_summarizer.LATENCY_BEGIN,
            "old latency",
            docs_summarizer.LATENCY_END,
            docs_summarizer.THROUGHPUT_BEGIN,
            "old throughput",
            docs_summarizer.THROUGHPUT_END,
            docs_summarizer.PROVENANCE_BEGIN,
            "old provenance",
            docs_summarizer.PROVENANCE_END,
        )
    )
    document_path = tmp_path / "benchmarks_core_runtime.mdx"
    document_path.write_text(document, encoding="utf-8")

    docs_summarizer.update_doc(document_path, "new latency", "new throughput", provenance)

    # Key assertion: one update refreshes tables and artifact-derived provenance together.
    updated = document_path.read_text(encoding="utf-8")
    assert "old provenance" not in updated
    assert "1970-01-01" in updated
    assert "`docs-core-runtime`; 7 repetitions" in updated
    assert "48/48 rows passed" in updated
    assert "Test GPU (driver 1.2.3)" in updated
    assert "new latency" in updated
    assert "new throughput" in updated
    assert "8 selected physical cores" in provenance
    assert "topology not recorded" not in provenance

    # Key assertion: a selected CPU that no artifact describes is named, not described, and the
    # topology wording covers only the recorded CPUs.
    native_manifest = json.loads((native_dir / "manifest.json").read_text(encoding="utf-8"))
    complete_topology = native_manifest["cpu_topology"]
    native_manifest["cpu_topology"] = [item for item in complete_topology if item["cpu"] != 7]
    write_json(native_dir / "manifest.json", native_manifest)
    partial = docs_summarizer.render_provenance(suite_dir, manifest, by_id)
    assert "recorded topology covers 7 physical cores" in partial
    assert "selected physical cores" not in partial
    assert "governor reported for CPUs with recorded topology" in partial
    assert "topology not recorded for CPU 7 |" in partial
    assert "support threads" not in partial

    # Key assertion: an extra selected CPU that no row uses is reported without inventing a role.
    native_manifest["cpu_topology"] = complete_topology
    write_json(native_dir / "manifest.json", native_manifest)
    extra_cpu = docs_summarizer.render_provenance(
        suite_dir, dict(manifest, cpus=[*manifest["cpus"], 8]), by_id
    )
    assert "recorded topology covers 8 physical cores" in extra_cpu
    assert "topology not recorded for CPU 8 |" in extra_cpu
    assert "support threads" not in extra_cpu

    # Key assertion: a pinned CPU outside the selection is still rejected.
    native_manifest["cpu_topology"].append(dict(native_manifest["cpu_topology"][0], cpu=11))
    write_json(native_dir / "manifest.json", native_manifest)
    with pytest.raises(RuntimeError, match="outside the selection"):
        docs_summarizer.render_provenance(suite_dir, manifest, by_id)


def test_balanced_plan_reduces_repetitions_and_preserves_p99_sampling(
    tmp_path: Path,
) -> None:
    # Key setup: build the routine measurement plan for the two dominant suite groups.
    args = suite_args(tmp_path)
    args.preset = "balanced"
    args.repetitions = 3
    args.groups = ["holomq_interprocess", "channel", "graph_interprocess"]
    cases = runner.build_cases(args)
    by_id = {case.case_id: case for case in cases}

    # Key assertion: routine runs retain three trials and at least 1,000 latency samples.
    channel = by_id["channel_matrix_latency_rep000"].command
    assert channel[channel.index("--pod-samples") + 1] == "5000"
    assert channel[channel.index("--host-samples") + 1] == "5000"
    assert channel[channel.index("--device-samples") + 1] == "5000"
    fanout = by_id["graph_interprocess_1to2_latency"].command
    assert fanout[fanout.index("--samples") + 1] == "2000"
    assert fanout[fanout.index("--warmup") + 1] == "200"
    assert fanout[fanout.index("--repetitions") + 1] == "3"
    assert fanout[fanout.index("--session-timeout-ms") + 1] == "30000"
    assert len([case for case in cases if case.case_id.startswith("channel_matrix_latency")]) == 3
    for case_id in (
        "graph_interprocess_tensor_pinned_host_1to1_latency",
        "graph_interprocess_image_device_1to1_latency",
    ):
        command = by_id[case_id].command
        assert command[command.index("--bulk-work-profile") + 1] == "descriptor_only"
    holomq_device = by_id["holomq_interprocess_buffer_device_throughput"].command
    assert holomq_device[holomq_device.index("--device-sync") + 1] == "none"
    holomq_device_ready = by_id["holomq_interprocess_buffer_device_ready_throughput"].command
    assert holomq_device_ready[holomq_device_ready.index("--device-sync") + 1] == "ready"
    assert runner.BALANCED_MEASUREMENT_POLICY["percentiles"]["p99_minimum_samples"] == 1_000


def test_smoke_plan_records_the_counts_used_by_commands(tmp_path: Path) -> None:
    # Key setup: build smoke rows for inline, host-bulk, and device-bulk payload classes.
    args = suite_args(tmp_path)
    cases = runner.build_cases(args)
    by_id = {case.case_id: case for case in cases}

    # Key assertion: manifest policy and generated commands share one count contract.
    assert runner.measurement_policy(args) == runner.SMOKE_MEASUREMENT_POLICY
    inline = by_id["graph_interprocess_1to1_latency"].command
    host = by_id["graph_interprocess_tensor_pinned_host_1to1_latency"].command
    device = by_id["graph_interprocess_tensor_device_1to1_latency"].command
    assert inline[inline.index("--samples") + 1] == "1000"
    assert host[host.index("--samples") + 1] == "200"
    assert device[device.index("--samples") + 1] == "100"
    assert inline[inline.index("--warmup") + 1] == "100"
    assert host[host.index("--warmup") + 1] == "100"
    assert device[device.index("--warmup") + 1] == "100"


def test_production_graph_fanout_timeout_covers_effective_pacing(tmp_path: Path) -> None:
    # Key setup: build the production plan whose 1:2 row raises pacing to five milliseconds.
    args = suite_args(tmp_path)
    args.preset = "production"
    args.repetitions = 7
    args.groups = ["graph_interprocess"]
    cases = runner.build_cases(args)
    by_id = {case.case_id: case for case in cases}

    # Key assertion: the worker and orchestration timeouts cover all paced samples.
    command = by_id["graph_interprocess_1to2_latency"].command
    assert command[command.index("--session-timeout-ms") + 1] == "110000"
    assert command[command.index("--process-timeout-sec") + 1] == "330"


def test_cpu_validation_rejects_sibling_threads(tmp_path: Path) -> None:
    # Key setup: model two logical CPUs backed by the same physical core.
    sysfs = tmp_path / "sysfs"
    for cpu in (2, 18):
        topology = sysfs / f"cpu{cpu}/topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text("0\n", encoding="utf-8")
        (topology / "core_id").write_text("2\n", encoding="utf-8")
        (topology / "thread_siblings_list").write_text("2,18\n", encoding="utf-8")

    # Key assertion: a manually supplied sibling pair fails before any benchmark starts.
    with pytest.raises(ValueError, match="sibling threads"):
        runner.validate_distinct_physical_cpus([2, 18], sysfs)


def test_cpu_validation_accepts_singleton_siblings_with_ambiguous_core_ids(tmp_path: Path) -> None:
    sysfs = tmp_path / "sysfs"
    for cpu in range(8):
        topology = sysfs / f"cpu{cpu}/topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text("0\n", encoding="utf-8")
        (topology / "core_id").write_text("0\n", encoding="utf-8")
        (topology / "thread_siblings_list").write_text(f"{cpu}\n", encoding="utf-8")

    runner.validate_distinct_physical_cpus(list(range(8)), sysfs)


def test_dry_run_writes_replayable_manifest_without_binaries(tmp_path: Path) -> None:
    # Key setup: request one group from an empty build tree in dry-run mode.
    output_dir = tmp_path / "dry-run"
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_DIR / "run_transport_benchmark_suite.py"),
            "--build-dir",
            str(tmp_path / "missing-build"),
            "--output-dir",
            str(output_dir),
            "--groups",
            "graph_interprocess",
            "--no-build",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # Key assertion: planning records exact commands while deliberately doing no execution.
    manifest = json.loads((output_dir / "suite_manifest.json").read_text(encoding="utf-8"))
    assert completed.stdout.startswith("planned 28 cases")
    assert len(manifest["cases"]) == 28
    assert {case["status"] for case in manifest["cases"]} == {"planned"}
    assert manifest["scenario_identity_contract"] == runner.scenario_identity_contract()
    assert manifest["workload_geometry_policy"] == runner.WORKLOAD_GEOMETRY_POLICY
    commands = (output_dir / "commands.sh").read_text(encoding="utf-8")
    assert "run_runtime_graph_interprocess_benchmark.py" in commands
    assert "--consumer-cpus" not in commands


def test_manifest_records_actionable_failure_and_skip_summary(tmp_path: Path) -> None:
    # Key setup: model one case in every suite status before writing the manifest.
    args = suite_args(tmp_path)
    args.created_unix_seconds = 123.0
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    cases = [
        runner.BenchmarkCase(
            case_id="planned-case",
            layer="graph",
            domain="intraprocess",
            workload="latency",
            result_kind="runtime_graph",
            output=output_dir / "planned/summary.json",
            command=["planned"],
        ),
        runner.BenchmarkCase(
            case_id="passed-case",
            layer="graph",
            domain="intraprocess",
            workload="latency",
            result_kind="runtime_graph",
            output=output_dir / "passed/summary.json",
            command=["passed"],
            status="passed",
            returncode=0,
        ),
        runner.BenchmarkCase(
            case_id="failed-case",
            layer="graph",
            domain="interprocess",
            workload="latency",
            result_kind="runtime_graph_interprocess",
            output=output_dir / "failed/summary.json",
            command=["failed"],
            status="failed",
            returncode=13,
        ),
        runner.BenchmarkCase(
            case_id="skipped-case",
            layer="graph",
            domain="interprocess",
            workload="saturated",
            result_kind="runtime_graph_interprocess",
            output=output_dir / "skipped/summary.json",
            command=[],
            status="skipped",
            returncode=77,
            required_cpus=8,
            skip_reason="topology requires 8 distinct physical CPUs; only 6 selected",
        ),
    ]

    runner.write_manifest(args, cases, output_dir, None)

    # Key assertion: automation can identify terminal failures and their exact replay artifacts.
    manifest = json.loads((output_dir / "suite_manifest.json").read_text(encoding="utf-8"))
    result = manifest["result_summary"]
    assert result["outcome"] == "failed"
    assert result["summarizer_returncode"] is None
    assert result["status_counts"] == {
        "planned": 1,
        "passed": 1,
        "failed": 1,
        "skipped": 1,
    }
    assert result["failed_cases"] == [
        {
            "case_id": "failed-case",
            "returncode": 13,
            "stdout_log": "logs/failed-case.stdout.log",
            "stderr_log": "logs/failed-case.stderr.log",
        }
    ]
    assert result["skipped_cases"] == [
        {
            "case_id": "skipped-case",
            "returncode": 77,
            "required_cpus": 8,
            "reason": "topology requires 8 distinct physical CPUs; only 6 selected",
        }
    ]
    assert runner.failure_message(cases[2], output_dir).endswith(
        f"failed-case: exit code 13; stderr: {output_dir}/logs/failed-case.stderr.log"
    )
    terminal_cases = [case for case in cases if case.status != "planned"]
    assert runner.suite_result_summary(terminal_cases, 0)["outcome"] == "failed"
    terminal_cases = [case for case in terminal_cases if case.status != "failed"]
    assert runner.suite_result_summary(terminal_cases, 0)["outcome"] == "passed"
    assert runner.suite_result_summary(terminal_cases, 2)["outcome"] == "failed"


def test_resume_preserves_completed_cases_and_updates_pending_commands(tmp_path: Path) -> None:
    # Key setup: retain one passed case from a compatible suite manifest.
    args = suite_args(tmp_path)
    args.groups = ["graph_interprocess"]
    cases = runner.build_cases(args)
    manifest = {
        "schema": runner.SUITE_SCHEMA,
        "preset": args.preset,
        "repetitions": args.repetitions,
        "latency_period_us": args.latency_period_us,
        "tensor_bytes": args.tensor_bytes,
        "groups": args.groups,
        "cpus": args.cpus,
        "build_dir": str(args.build_dir.resolve()),
        "cases": [case.manifest_record(args.output_dir.resolve()) for case in cases],
    }
    manifest["cases"][0]["status"] = "passed"
    manifest["cases"][0]["returncode"] = 0
    manifest["cases"][0]["command"] = ["recorded", "passed", "command"]
    manifest["cases"][1]["status"] = "failed"
    manifest["cases"][1]["returncode"] = 1
    manifest["cases"][1]["command"] = ["stale", "failed", "command"]
    replanned_command = list(cases[1].command)

    runner.restore_completed_cases(args, cases, manifest)

    # Key assertion: resume skips completed work but replans failed or unstarted cases.
    assert cases[0].status == "passed"
    assert cases[0].returncode == 0
    assert cases[0].command == ["recorded", "passed", "command"]
    assert cases[1].status == "planned"
    assert cases[1].returncode is None
    assert cases[1].command == replanned_command
    assert all(case.status == "planned" for case in cases[2:])


def test_resume_requires_binaries_only_for_pending_case_groups(tmp_path: Path) -> None:
    # Key setup: mark every Graph case complete while leaving one Channel case pending.
    args = suite_args(tmp_path)
    args.groups = ["channel", "graph_interprocess"]
    cases = runner.build_cases(args)
    for case in cases:
        if runner.case_group(case) == "graph_interprocess":
            case.status = "passed"

    # Key assertion: completed groups no longer contribute a binary prerequisite.
    assert runner.pending_case_groups(cases) == {"channel"}


def test_unpinned_graph_fanout_omits_list_cpu_arguments(tmp_path: Path) -> None:
    # Key setup: build the intraprocess Graph plan without controlled CPU placement.
    args = suite_args(tmp_path)
    args.groups = ["graph_intraprocess"]
    args.cpus = None
    cases = runner.build_cases(args)
    fanout = next(case for case in cases if case.case_id == "graph_intraprocess_pod_1to2_latency")

    # Key assertion: runner defaults represent unpinned execution without invalid -1 lists.
    assert "--sink-cpus" not in fanout.command
    assert "--source-cpu" not in fanout.command


def test_latency_contract_records_known_minimum_period_overrides() -> None:
    # Key setup: model the two rows whose effective period exceeds the common request.
    device_row = {
        "workload": "latency",
        "publisher_pacing": "spin",
        "effective_period_us": 500,
        "layer": "holomq",
        "payload": "buffer_handle",
        "storage": "device",
    }
    fanout_row = {
        "workload": "latency",
        "publisher_pacing": "notification_driver_sleep",
        "effective_period_us": 5_000,
        "layer": "graph",
        "payload": "pod",
        "storage": "inline",
    }

    # Key assertion: normalized rows explain both exceptions instead of appearing comparable.
    summarizer.apply_latency_contract(device_row, {"case_id": "holomq-device"}, 200)
    summarizer.apply_latency_contract(
        fanout_row, {"case_id": "graph_interprocess_1to2_latency"}, 200
    )
    assert device_row["pacing_period_reason"] == "device_ready_release_headroom"
    assert fanout_row["pacing_period_reason"] == "distributed_graph_1to2_flow_control_minimum"
    assert device_row["pacing_period_policy"] == "minimum_period_override"
    assert fanout_row["pacing_period_policy"] == "minimum_period_override"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_scenario_identity_distinguishes_holomq_buffer_handle_delivery(
    tmp_path: Path,
) -> None:
    # Key setup: model a device-backed HoloMQ endpoint with explicit ready-event ordering.
    result_path = tmp_path / "summary.json"
    write_json(
        result_path,
        {
            "benchmark": "holomq_buffer_handle_transport",
            "workload": "throughput",
            "payload": "buffer_handle",
            "storage": "device",
            "device_sync": "ready",
            "application_payload_bytes": 1_048_576,
            "inline_transport_bytes": 64,
            "external_buffer_bytes": 1_048_576,
            "payload_write_profile": "reused_preinitialized_contents",
            "bulk_work_profile": "descriptor_only",
            "wake": {
                "event_wake": False,
                "busy_poll": False,
                "receive_polling": True,
            },
            "clock": {
                "source": "clock_gettime(CLOCK_MONOTONIC)",
                "steady": True,
                "scope": "same_host_cross_process",
            },
            "runtime_support_threads": "none",
            "payload_validation": "final",
            "validation_timing": "after_timed_operations",
            "subscriber_count": 1,
            "queue_depth": 8,
            "pool_slots": 10,
            "samples": 1_000,
            "warmup": 100,
            "metrics": {
                "completed_messages_per_second": {"median": 20_000.0},
                "completed_deliveries_per_second": {"median": 20_000.0},
                "buffer_acquire_p50_ns": {"median": 75.0},
                "buffer_import_p50_ns": {"median": 125.0},
                "final_validation_ns": {"median": 2_500.0},
            },
            "runs": [
                {
                    "completed_messages_per_second": 20_000.0,
                    "publication_counts": {
                        "scope": "measured_plus_warmup",
                        "attempted": 1_101,
                        "accepted": 1_100,
                        "backpressured": 1,
                        "indeterminate": 0,
                        "completed": 1_100,
                    },
                }
            ],
        },
    )

    row = summarizer.complete_scenario_dimensions(summarizer.holomq_interprocess_row(result_path))

    # Key assertion: the normalized identity does not collapse BufferHandle into intraprocess.
    assert row["delivery_path"] == "holomq_buffer_handle"
    assert row["synchronization"] == "cuda_ready_event"
    assert row["scenario_payload"] == "raw_buffer_1048576"
    assert row["execution_model"] == "concurrent"
    assert row["pool_acquisition_p50_ns"] == 75.0
    assert row["import_materialization_p50_ns"] == 125.0
    assert row["final_validation_ns"] == 2_500.0
    assert row["cuda_enqueue_p50_ns"] is None
    assert row["deferred_release_p50_ns"] is None


def test_graph_interprocess_summary_preserves_image_transport_identity(
    tmp_path: Path,
) -> None:
    # Key setup: model one device-backed Image result using worker-emitted geometry.
    result_path = tmp_path / "summary.json"
    write_json(
        result_path,
        {
            "benchmark": "runtime_graph_interprocess_image_1to1_device_descriptor_only",
            "payload": "image",
            "tensor_storage": "device",
            "workload": "latency",
            "topology": "fanout",
            "fanout": 1,
            "clock": {
                "source": "clock_gettime(CLOCK_MONOTONIC)",
                "steady": True,
                "scope": "same_host_cross_process",
            },
            "wake": {
                "event_wake": True,
                "busy_poll": False,
                "receive_polling": False,
            },
            "runtime_support_threads": "controller,channel_pump",
            "payload_validation": "final",
            "validation_timing": "after_timed_operations",
            "slowest_sink_end_to_end_p50_ns": {"p50": 8_000.0},
            "slowest_sink_end_to_end_p99_ns": {"p50": 12_000.0},
            "slowest_sink_end_to_end_p999_ns": None,
            "producer": [
                {
                    "geometry": {
                        "samples": 1_000,
                        "warmup": 100,
                        "queue_depth": 1,
                        "notification_capacity": 1,
                        "publication_backpressure_capacity": 1,
                        "pool_slots": 6,
                        "release_capacity": 6,
                        "publisher_period_us": 200,
                        "application_payload_bytes": 1 << 20,
                        "inline_transport_bytes": 856,
                        "external_buffer_bytes": 1 << 20,
                        "payload_write_profile": "preinitialized_immutable_backing",
                        "bulk_work_profile": "descriptor_only",
                    },
                    "counts": {"publications": 1_100, "backpressured_emits": 0},
                }
            ],
            "consumers": [[{"timing": {"end_to_end_ns": {"p50": 8_000.0, "p99": 12_000.0}}}]],
        },
    )

    row = summarizer.complete_scenario_dimensions(summarizer.graph_interprocess_row(result_path))

    # Key assertion: normalization retains the Tensor-backed schema's true transport dimensions.
    assert row["payload"] == "image"
    assert row["storage"] == "device"
    assert row["application_payload_bytes"] == 1 << 20
    assert row["inline_transport_bytes"] == 856
    assert row["external_buffer_bytes"] == 1 << 20
    assert row["pool_slots"] == 6
    assert row["release_capacity"] == 6
    assert row["delivery_path"] == "holomq_buffer_handle"
    assert row["synchronization"] == "cuda_stream_ordered"


def pinned_host_graph_result(payload: str, writer: str | None) -> dict[str, object]:
    """Return one two-process pinned-host result whose workers report `writer`, if any."""
    geometry: dict[str, object] = {
        "samples": 1_000,
        "warmup": 100,
        "queue_depth": 2,
        "notification_capacity": 2,
        "publication_backpressure_capacity": 2,
        "pool_slots": 7,
        "release_capacity": 7,
        "publisher_period_us": None,
        "application_payload_bytes": 1 << 20,
        "inline_transport_bytes": 856,
        "external_buffer_bytes": 1 << 20,
        "payload_write_profile": "preinitialized_immutable_backing",
        "bulk_work_profile": "descriptor_only",
    }
    result: dict[str, object] = {
        "benchmark": f"runtime_graph_interprocess_{payload}_1to1_pinned_host_descriptor_only",
        "payload": payload,
        "tensor_storage": "pinned_host",
        "workload": "saturated",
        "topology": "fanout",
        "fanout": 1,
        "clock": {
            "source": "clock_gettime(CLOCK_MONOTONIC)",
            "steady": True,
            "scope": "same_host_cross_process",
        },
        "wake": {
            "event_wake": True,
            "busy_poll": False,
            "receive_polling": False,
        },
        "runtime_support_threads": "controller,channel_pump",
        "payload_validation": "none",
        "validation_timing": "after_timed_operations",
        "producer": [
            {
                "geometry": geometry,
                "counts": {"publications": 1_100, "backpressured_emits": 0},
                "timing": {"messages_per_second": 50_000.0},
            }
        ],
        "consumers": [[{"timing": {"messages_per_second": 50_000.0}}]],
    }
    if writer is not None:
        result["pinned_host_writer"] = writer
        geometry["pinned_host_writer"] = writer
    return result


@pytest.mark.parametrize(
    ("writer", "synchronization"), [("host", "none"), ("cuda", "cuda_stream_ordered")]
)
def test_graph_interprocess_summary_orders_pinned_host_rows_by_their_writer(
    tmp_path: Path, writer: str, synchronization: str
) -> None:
    # Key setup: model one pinned-host Image result whose workers report their loan writer.
    result_path = tmp_path / "summary.json"
    write_json(result_path, pinned_host_graph_result("image", writer))

    row = summarizer.complete_scenario_dimensions(summarizer.graph_interprocess_row(result_path))

    # Key assertion: only the CUDA writer gives a pinned-host row a producer fence, so the two
    # writers never share one scenario identity.
    assert row["storage"] == "pinned_host"
    assert row["synchronization"] == synchronization
    assert row["pinned_host_writer"] == writer


def test_graph_interprocess_scenario_key_separates_pinned_host_writers(tmp_path: Path) -> None:
    # Key setup: normalize the same pinned-host Tensor scenario once per writer.
    result_path = tmp_path / "summary.json"
    identities = {}
    for writer in ("host", "cuda"):
        write_json(result_path, pinned_host_graph_result("tensor", writer))
        row = summarizer.complete_scenario_dimensions(
            summarizer.graph_interprocess_row(result_path)
        )
        identities[writer] = summarizer.make_scenario_identity(row)
        assert identities[writer]["pinned_host_writer"] == writer
    keys = {writer: summarizer.scenario_identity_key(value) for writer, value in identities.items()}
    assert keys["host"] != keys["cuda"]

    # Key assertion: before the writer was recorded, a CUDA-written Tensor row normalized with no
    # synchronization and no writer, which is the current host identity without its writer. That
    # stored key must not match either current row.
    legacy_identity = dict(identities["host"])
    legacy_identity.pop("pinned_host_writer")
    assert summarizer.scenario_identity_key(legacy_identity) not in keys.values()
    assert "pinned_host_writer" in summarizer.scenario_identity_contract()["fields"]


@pytest.mark.parametrize("payload", ["tensor", "image", "multi_tensor_loan"])
def test_graph_interprocess_summary_rejects_pinned_host_result_without_its_writer(
    tmp_path: Path, payload: str
) -> None:
    result_path = tmp_path / "summary.json"
    write_json(result_path, pinned_host_graph_result(payload, None))

    # Key assertion: an older result that omits its writer is rejected, never treated as host.
    with pytest.raises(RuntimeError, match="which writer filled its pinned-host loans"):
        summarizer.graph_interprocess_row(result_path)


def test_summarizer_rejects_a_suite_from_an_older_scenario_identity(tmp_path: Path) -> None:
    contract = summarizer.scenario_identity_contract()
    older_contract = {
        **contract,
        "schema": "holoscan.transport-benchmark-scenario/v4",
        "fields": [field for field in contract["fields"] if field != "pinned_host_writer"],
    }
    write_json(
        tmp_path / "suite_manifest.json",
        {"scenario_identity_contract": older_contract, "latency_period_us": 1_000, "cases": []},
    )

    # Key assertion: a suite recorded under an older identity is not renormalized into the
    # current one, so its rows cannot silently gain or lose a dimension.
    with pytest.raises(RuntimeError, match="unsupported scenario identity contract"):
        summarizer.collect_rows(tmp_path)


def test_graph_intraprocess_summarizer_requires_requested_operator_lanes(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "graph-summary.json"
    result = {
        "benchmark": "runtime_graph_tensor_1to1",
        "workload": "latency",
        "source_trigger": "clock",
        "scenario": "tensor",
        "process_local_inline_delivery": "direct_retained",
        "clock": {
            "source": "std::chrono::steady_clock",
            "steady": True,
            "scope": "single_process",
        },
        "tensor": {"storage": "device"},
        "geometry": {
            "fanout": 1,
            "application_payload_bytes": 16,
            "inline_transport_bytes": 0,
            "external_buffer_bytes": 16,
            "payload_write_profile": "full_payload_per_sample",
            "bulk_work_profile": "full_write",
            "samples": 1,
            "warmup": 0,
            "allocation_probe_samples": 0,
            "publisher_period_us": 200,
        },
        "wake": {"event_wake": False, "busy_poll": False, "receive_polling": False},
        "runtime_support_threads": "controller",
        "placement": {"requested_operator_lanes": "shared"},
        "metrics": {
            "publication_to_receive_p50_ns_across_runs": {"median": 4_000.0},
            "publication_to_receive_p99_ns_across_runs": {"median": 8_000.0},
            "final_validation_ns_across_runs": {"median": 1_500.0},
        },
        "runs": [
            {
                "publication_to_receive_p50_ns": 4_000.0,
                "counts": {"publications": 1, "backpressured_emits": 0},
            }
        ],
        "validation": {"mode": "final", "timing": "after_timed_operations"},
        "timestamp_instrumentation": {"pair_overhead_p50_ns_across_runs": {"median": 21.0}},
    }
    write_json(result_path, result)

    row = summarizer.graph_intraprocess_row(result_path, "graph_intraprocess_tensor_device_latency")
    assert row["execution_model"] == "shared_lane"
    assert row["input_access"] == "not_applicable"

    # Key assertion: owning and borrowed ImageT sinks are distinct scenarios, so they never merge.
    image_rows = []
    for accessor in ("receive", "selected_sample"):
        image_result = dict(result, scenario="image", image_input_accessor=accessor)
        image_result["benchmark"] = "runtime_graph_image_1to1"
        write_json(result_path, image_result)
        image_row = summarizer.graph_intraprocess_row(
            result_path, f"graph_intraprocess_image_{accessor}"
        )
        assert image_row["input_access"] == accessor
        image_rows.append(image_row)
    grouping_keys = {
        tuple(row.get(field) for field in summarizer.IDENTITY_FIELDS) for row in image_rows
    }
    assert len(grouping_keys) == 2
    assert "input_access" in summarizer.scenario_identity_contract()["fields"]
    write_json(result_path, result)

    for placement in ({}, {"operator_lanes": "shared"}, {"requested_operator_lanes": 1}):
        result["placement"] = placement
        write_json(result_path, result)
        with pytest.raises(RuntimeError, match="requested_operator_lanes"):
            summarizer.graph_intraprocess_row(
                result_path, "graph_intraprocess_tensor_device_latency"
            )


def test_summarizer_normalizes_all_existing_result_schemas(tmp_path: Path) -> None:
    # Key setup: create one minimal valid artifact for each existing benchmark result producer.
    suite_dir = tmp_path / "suite"
    write_json(
        suite_dir / "holomq-intra.json",
        {
            "benchmarks": [
                {
                    "name": "BM_PubSubPOD_OneToOne_Blocking/real_time",
                    "real_time": 100.0,
                    "time_unit": "ns",
                    "items_per_second": 1_000_000.0,
                    "iterations": 100,
                    "subscribers": 1,
                    "event_wake": 0,
                    "busy_poll": 0,
                    "receive_polling": 1,
                },
                {
                    "name": "BM_PubSubPOD_OneToOne_Blocking/real_time",
                    "real_time": 90.0,
                    "time_unit": "ns",
                    "items_per_second": 1_200_000.0,
                    "iterations": 100,
                    "subscribers": 1,
                    "event_wake": 0,
                    "busy_poll": 0,
                    "receive_polling": 1,
                },
                {
                    "name": "BM_PubSubFixed_PollingRoundTrip<64>/real_time",
                    "real_time": 800.0,
                    "time_unit": "ns",
                    "event_wake": 0,
                    "busy_poll": 0,
                    "receive_polling": 1,
                    "iterations": 100,
                },
                {
                    "name": "BM_PubSubPOD_ConcurrentWaitSet_Latency/manual_time",
                    "real_time": 1_500.0,
                    "time_unit": "ns",
                    "latency_workload": 1,
                    "event_wake": 1,
                    "busy_poll": 0,
                    "receive_polling": 0,
                    "subscribers": 1,
                    "iterations": 100,
                },
            ]
        },
    )
    write_json(
        suite_dir / "holomq-ipc/summary.json",
        {
            "benchmark": "holomq_inline_transport",
            "workload": "latency",
            "payload": "inline",
            "storage": "inline",
            "payload_bytes": 64,
            "application_payload_bytes": 64,
            "inline_transport_bytes": 64,
            "external_buffer_bytes": 0,
            "payload_write_profile": "full_payload_per_sample",
            "bulk_work_profile": "not_applicable",
            "wake": {
                "event_wake": False,
                "busy_poll": False,
                "receive_polling": True,
            },
            "runtime_support_threads": "none",
            "samples": 1_000,
            "warmup": 100,
            "publisher_period_us": 200,
            "publisher_pacing": "spin",
            "clock": {
                "source": "clock_gettime(CLOCK_MONOTONIC)",
                "steady": True,
                "scope": "same_host_cross_process",
            },
            "subscriber_count": 1,
            "metrics": {
                "worst_subscriber_latency_p50_ns": {"median": 2_000.0},
                "worst_subscriber_latency_p99_ns": {"median": 4_000.0},
            },
            "runs": [
                {
                    "worst_subscriber_latency_p50_ns": 2_000.0,
                    "publication_counts": {
                        "scope": "measured_plus_warmup",
                        "attempted": 1_100,
                        "accepted": 1_100,
                        "backpressured": 0,
                        "indeterminate": 0,
                        "completed": 1_100,
                    },
                }
            ],
        },
    )
    write_json(
        suite_dir / "channel/pod.json",
        {
            "benchmark": "channel_interprocess_pod_1to2",
            "domain": "interprocess",
            "publisher": {
                "payload": "pod",
                "transport": "inline",
                "workload": "throughput",
                "payload_bytes": 64,
                "application_payload_bytes": 64,
                "inline_transport_bytes": 64,
                "external_buffer_bytes": 0,
                "payload_write_profile": "full_payload_per_sample",
                "bulk_work_profile": "not_applicable",
                "publisher_period_us": 0,
                "publisher_pacing": "sleep",
                "samples": 1_000,
                "warmup": 100,
                "subscriber_queue_depth": 64,
                "failed_emits": 0,
                "clock": {
                    "source": "clock_gettime(CLOCK_MONOTONIC)",
                    "steady": True,
                    "scope": "same_host_cross_process",
                },
                "wake": {
                    "event_wake": False,
                    "busy_poll": False,
                    "receive_polling": True,
                },
                # Tensor-only options must not leak into normalized POD dimensions.
                "tensor_storage": "pinned_host",
                "tensor_bytes": 1_048_576,
            },
            "subscribers": [{}, {}],
            "fanout": {
                "completed_logical_messages_per_second": 500_000.0,
                "completed_deliveries_per_second": 1_000_000.0,
            },
        },
    )
    write_json(
        suite_dir / "channel/tensor-sidecar.json",
        {
            "benchmark": "channel_intraprocess_sidecar_tensor_pinned_host_1to1",
            "domain": "intraprocess_sidecar",
            "publisher": {
                "payload": "tensor",
                "transport": "sidecar",
                "tensor_storage": "pinned_host",
                "async_cuda_work": False,
                "workload": "throughput",
                "application_payload_bytes": 1_048_576,
                "inline_transport_bytes": 128,
                "external_buffer_bytes": 1_048_576,
                "payload_write_profile": "reused_preinitialized_contents",
                "bulk_work_profile": "descriptor_only",
                "samples": 200,
                "warmup": 100,
                "subscriber_queue_depth": 64,
                "failed_emits": 0,
                "pool_slots": 256,
                "release_capacity": 256,
                "max_fanout_release_streams": 1,
                "payload_validation": "final",
                "validation_timing": "after_timed_operations",
                "clock": {
                    "source": "std::chrono::steady_clock",
                    "steady": True,
                    "scope": "process_local",
                },
                "wake": {
                    "event_wake": False,
                    "busy_poll": False,
                    "receive_polling": True,
                },
            },
            "subscribers": [
                {
                    "channel_receive_p50_ns": 350,
                    "final_validation_ns": 1_200,
                }
            ],
            "fanout": {
                "completed_logical_messages_per_second": 100_000.0,
                "completed_deliveries_per_second": 100_000.0,
            },
        },
    )
    write_json(
        suite_dir / "graph-intra/summary.json",
        {
            "benchmark": "runtime_graph_tensor_1to1",
            "workload": "latency",
            "source_trigger": "clock",
            "scenario": "tensor",
            "process_local_inline_delivery": "direct_retained",
            "clock": {
                "source": "std::chrono::steady_clock",
                "steady": True,
                "scope": "single_process",
            },
            "timestamp_instrumentation": {"pair_overhead_p50_ns_across_runs": {"median": 21.0}},
            "geometry": {
                "samples": 1_000,
                "warmup": 100,
                "fanout": 1,
                "payload_bytes": 1_048_576,
                "application_payload_bytes": 1_048_576,
                "inline_transport_bytes": 0,
                "external_buffer_bytes": 1_048_576,
                "queue_depth": 1,
                "notification_capacity": 0,
                "pool_slots": 4,
                "release_capacity": 4,
                "publication_backpressure_capacity": 1,
                "max_fanout_release_streams": 1,
                "allocation_probe_samples": 1,
                "payload_write_profile": "full_payload_per_sample",
                "bulk_work_profile": "full_write",
                "publisher_period_us": 200,
            },
            "wake": {
                "event_wake": False,
                "busy_poll": False,
                "receive_polling": False,
            },
            "runtime_support_threads": "controller",
            "placement": {"requested_operator_lanes": "shared"},
            "metrics": {
                "latency_p50_ns_across_runs": {"median": 5_000.0},
                "latency_p99_ns_across_runs": {"median": 9_000.0},
                "publication_to_receive_p50_ns_across_runs": {"median": 4_000.0},
                "publication_to_receive_p99_ns_across_runs": {"median": 8_000.0},
                "final_validation_ns_across_runs": {"median": 1_500.0},
            },
            "runs": [
                {
                    "publication_to_receive_p50_ns": 4_000.0,
                    "counts": {
                        "publications": 1_101,
                        "backpressured_emits": 0,
                    },
                }
            ],
            "validation": {
                "mode": "final",
                "timing": "after_timed_operations",
            },
        },
    )
    write_json(
        suite_dir / "graph-intra-schema/summary.json",
        {
            "benchmark": "runtime_graph_schema_1to2",
            "workload": "latency",
            "source_trigger": "clock",
            "scenario": "schema",
            "process_local_inline_delivery": "holomq",
            "clock": {
                "source": "std::chrono::steady_clock",
                "steady": True,
                "scope": "single_process",
            },
            "timestamp_instrumentation": {"pair_overhead_p50_ns_across_runs": {"median": 22.0}},
            "geometry": {
                "samples": 1_000,
                "warmup": 100,
                "fanout": 2,
                "application_payload_bytes": 16,
                "inline_transport_bytes": None,
                "external_buffer_bytes": 0,
                "queue_depth": 1,
                "notification_capacity": 0,
                "publication_backpressure_capacity": 1,
                "allocation_probe_samples": 1,
                "payload_write_profile": "full_payload_per_sample",
                "bulk_work_profile": "not_applicable",
                "publisher_period_us": 200,
            },
            "wake": {
                "event_wake": False,
                "busy_poll": False,
                "receive_polling": False,
            },
            "runtime_support_threads": "controller",
            "placement": {"requested_operator_lanes": "per_sink"},
            "metrics": {
                "latency_p50_ns_across_runs": {"median": 6_000.0},
                "latency_p99_ns_across_runs": {"median": 10_000.0},
                "publication_to_receive_p50_ns_across_runs": {"median": 5_000.0},
                "publication_to_receive_p99_ns_across_runs": {"median": 9_000.0},
            },
            "runs": [
                {
                    "publication_to_receive_p50_ns": 5_000.0,
                    "counts": {
                        "publications": 1_101,
                        "backpressured_emits": 0,
                    },
                }
            ],
            "validation": {
                "mode": "final",
                "timing": "after_timed_operations",
            },
        },
    )
    write_json(
        suite_dir / "graph-ipc/summary.json",
        {
            "benchmark": "runtime_graph_interprocess_pod_1to1",
            "workload": "latency",
            "fanout": 1,
            "clock": {
                "source": "clock_gettime(CLOCK_MONOTONIC)",
                "steady": True,
                "scope": "same_host_cross_process",
            },
            "timestamp_instrumentation": {
                "pair_overhead_p50_ns_across_processes_and_runs": {"median": 25.0}
            },
            "wake": {
                "event_wake": True,
                "busy_poll": False,
                "receive_polling": False,
            },
            "runtime_support_threads": "controller,channel_pump",
            "payload_validation": "final",
            "validation_timing": "after_timed_operations",
            "slowest_sink_end_to_end_p50_ns": {"p50": 7_000.0},
            "slowest_sink_end_to_end_p99_ns": {"median": 13_000.0},
            "slowest_sink_end_to_end_p999_ns": None,
            "producer": [
                {
                    "geometry": {
                        "samples": 1_000,
                        "warmup": 100,
                        "application_payload_bytes": 64,
                        "inline_transport_bytes": 64,
                        "external_buffer_bytes": 0,
                        "payload_write_profile": "value_fill",
                        "bulk_work_profile": "not_applicable",
                        "queue_depth": 1,
                        "notification_capacity": 1,
                        "publication_backpressure_capacity": 1,
                        "publisher_period_us": 200,
                    },
                    "counts": {
                        "publications": 1_100,
                        "backpressured_emits": 0,
                    },
                }
            ],
            "consumers": [
                [{"timing": {"end_to_end_ns": {"p50": 7_000.0, "p99": 12_000.0}}}],
                [{"timing": {"end_to_end_ns": {"p50": 8_000.0, "p99": 14_000.0}}}],
            ],
        },
    )
    manifest_cases = [
        ("gb", "google_benchmark", "holomq-intra.json"),
        ("hmq", "holomq_interprocess", "holomq-ipc/summary.json"),
        ("channel", "channel_matrix", "channel"),
        (
            "graph_intraprocess_tensor_device_latency",
            "graph_intraprocess",
            "graph-intra/summary.json",
        ),
        (
            "graph_intraprocess_schema_1to2_latency",
            "graph_intraprocess",
            "graph-intra-schema/summary.json",
        ),
        ("graph-ipc", "graph_interprocess", "graph-ipc/summary.json"),
    ]
    write_json(
        suite_dir / "suite_manifest.json",
        {
            "schema": runner.SUITE_SCHEMA,
            "preset": "smoke",
            "latency_period_us": 200,
            "scenario_identity_contract": runner.scenario_identity_contract(),
            "workload_geometry_policy": runner.WORKLOAD_GEOMETRY_POLICY,
            "cases": [
                {
                    "case_id": case_id,
                    "result_kind": kind,
                    "output": output,
                    "status": "passed",
                    "command": ["benchmark", "--case", case_id],
                }
                for case_id, kind, output in manifest_cases
            ],
        },
    )

    _, rows = summarizer.collect_rows(suite_dir)
    by_benchmark = {row["benchmark"]: row for row in rows}

    # Key assertion: rates and percentiles retain their native meaning across schemas.
    pod_row = by_benchmark["BM_PubSubPOD_OneToOne_Blocking"]
    assert pod_row["logical_messages_per_second"] == 1_100_000.0
    assert pod_row["repetitions"] == 2
    assert pod_row["primary_metric_min"] == 1_000_000.0
    assert pod_row["primary_metric_median"] == 1_100_000.0
    assert pod_row["primary_metric_max"] == 1_200_000.0
    assert pod_row["primary_metric_coefficient_of_variation"] is not None
    assert by_benchmark["BM_PubSubPOD_OneToOne_Blocking"]["application_payload_bytes"] == 64
    assert by_benchmark["BM_PubSubPOD_OneToOne_Blocking"]["inline_transport_bytes"] == 64
    assert by_benchmark["BM_PubSubPOD_OneToOne_Blocking"]["external_buffer_bytes"] == 0
    assert by_benchmark["BM_PubSubPOD_OneToOne_Blocking"]["receive_polling"]
    assert by_benchmark["BM_PubSubPOD_OneToOne_Blocking"]["execution_model"] == "serialized"
    assert by_benchmark["BM_PubSubPOD_OneToOne_Blocking"]["delivery_path"] == "holomq_inline"
    assert by_benchmark["BM_PubSubFixed_PollingRoundTrip<64>"]["mean_iteration_ns"] == 800.0
    assert by_benchmark["BM_PubSubPOD_ConcurrentWaitSet_Latency"]["mean_iteration_ns"] == 1_500.0
    assert by_benchmark["BM_PubSubPOD_ConcurrentWaitSet_Latency"]["event_wake"]
    assert by_benchmark["BM_PubSubPOD_ConcurrentWaitSet_Latency"]["execution_model"] == "concurrent"
    assert not by_benchmark["BM_PubSubPOD_ConcurrentWaitSet_Latency"]["receive_polling"]
    assert by_benchmark["holomq_inline_transport"]["latency_p99_ns"] == 4_000.0
    assert by_benchmark["holomq_inline_transport"]["publisher_pacing"] == "spin"
    assert by_benchmark["holomq_inline_transport"]["comparison_period_us"] == 200
    assert by_benchmark["holomq_inline_transport"]["effective_period_us"] == 200
    assert (
        by_benchmark["holomq_inline_transport"]["pacing_period_policy"] == "common_requested_period"
    )
    assert by_benchmark["holomq_inline_transport"]["clock_scope"] == "same_host_cross_process"
    channel_pod = by_benchmark["channel_interprocess_pod_1to2"]
    assert channel_pod["aggregate_deliveries_per_second"] == 1_000_000.0
    assert channel_pod["storage"] == "inline"
    assert channel_pod["application_payload_bytes"] == 64
    assert channel_pod["inline_transport_bytes"] == 64
    assert channel_pod["external_buffer_bytes"] == 0
    assert channel_pod["payload_write_profile"] == "full_payload_per_sample"
    assert channel_pod["bulk_work_profile"] == "not_applicable"
    assert channel_pod["runtime_support_threads"] == "none"
    assert channel_pod["topology"] == "1to2"
    assert channel_pod["delivery_path"] == "holomq_inline"
    assert channel_pod["execution_model"] == "concurrent"
    tensor_sidecar = by_benchmark["channel_intraprocess_sidecar_tensor_pinned_host_1to1"]
    assert tensor_sidecar["delivery_path"] == "holomq_sidecar"
    assert tensor_sidecar["bulk_work_profile"] == "descriptor_only"
    assert tensor_sidecar["import_materialization_p50_ns"] == 350
    assert tensor_sidecar["final_validation_ns"] == 1_200
    assert tensor_sidecar["pool_acquisition_p50_ns"] is None
    assert tensor_sidecar["deferred_release_p50_ns"] is None
    graph_tensor = by_benchmark["runtime_graph_tensor_1to1"]
    assert graph_tensor["storage"] == "device"
    assert graph_tensor["application_payload_bytes"] == 1_048_576
    assert graph_tensor["inline_transport_bytes"] == 0
    assert graph_tensor["external_buffer_bytes"] == 1_048_576
    assert graph_tensor["publisher_pacing"] == "runtime_on_clock"
    assert graph_tensor["timestamp_read_pair_p50_ns"] == 21.0
    assert graph_tensor["delivery_path"] == "direct_retained"
    assert graph_tensor["execution_model"] == "shared_lane"
    assert graph_tensor["final_validation_ns"] == 1_500
    assert graph_tensor["pool_acquisition_p50_ns"] is None
    assert graph_tensor["import_materialization_p50_ns"] is None
    assert graph_tensor["cuda_enqueue_p50_ns"] is None
    assert graph_tensor["deferred_release_p50_ns"] is None
    assert graph_tensor["scenario_identity"] == {
        "scope": "operator_graph",
        "domain": "intraprocess",
        "topology": "1to1",
        "workload": "rate_limited_latency",
        "trigger": "runtime_on_clock",
        "payload": "tensor",
        "storage": "device",
        "bulk_work_profile": "full_write",
        "delivery": "direct_retained",
        "execution": "shared_lane",
        "input_access": "not_applicable",
        "synchronization": "cuda_stream_ordered",
        "pinned_host_writer": "not_applicable",
        "wake": "sleeping_waitset",
        "geometry": {
            "fanout": 1,
            "relay_count": 0,
            "queue_depth": 1,
            "notification_capacity": 0,
            "pool_slots": 4,
            "release_capacity": 4,
            "publication_backpressure_capacity": 1,
            "max_fanout_release_streams": 1,
            "process_count": 1,
            "application_payload_bytes": 1_048_576,
            "inline_transport_bytes": 0,
            "external_buffer_bytes": 1_048_576,
        },
    }
    assert by_benchmark["runtime_graph_tensor_1to1"]["publication_attempted_count"] == 1_101
    assert by_benchmark["runtime_graph_tensor_1to1"]["publication_completed_count"] == 1_101
    assert by_benchmark["runtime_graph_tensor_1to1"]["native_benchmark"] == (
        "runtime_graph_tensor_1to1"
    )
    assert by_benchmark["runtime_graph_tensor_1to1"]["native_runs"] == [
        {
            "manifest_case_id": "graph_intraprocess_tensor_device_latency",
            "artifact": "graph-intra/summary.json",
            "command": [
                "benchmark",
                "--case",
                "graph_intraprocess_tensor_device_latency",
            ],
        }
    ]
    assert by_benchmark["runtime_graph_schema_1to2"]["inline_transport_bytes"] is None
    assert by_benchmark["runtime_graph_schema_1to2"]["delivery_path"] == "holomq_inline"
    assert by_benchmark["runtime_graph_schema_1to2"]["execution_model"] == "per_subscriber_lane"
    assert by_benchmark["runtime_graph_interprocess_pod_1to1"]["execution_model"] == "separate_lane"
    assert by_benchmark["runtime_graph_interprocess_pod_1to1"]["application_payload_bytes"] == 64
    assert by_benchmark["runtime_graph_interprocess_pod_1to1"]["inline_transport_bytes"] == 64
    assert by_benchmark["runtime_graph_interprocess_pod_1to1"]["latency_p99_ns"] == 13_000.0
    assert (
        by_benchmark["runtime_graph_interprocess_pod_1to1"]["runtime_support_threads"]
        == "controller,channel_pump"
    )

    # Key assertion: human- and machine-readable outputs preserve the same identity and trace.
    summarizer.write_csv(suite_dir / "summary.csv", rows)
    summarizer.write_markdown(suite_dir / "summary.md", rows)
    csv_text = (suite_dir / "summary.csv").read_text(encoding="utf-8")
    markdown_text = (suite_dir / "summary.md").read_text(encoding="utf-8")
    assert "scenario_identity" in csv_text
    assert "direct_retained" in csv_text
    assert "Manifest cases" in markdown_text
