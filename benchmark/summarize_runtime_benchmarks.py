#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the public Core Runtime benchmark tables from one maintained suite run."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import summarize_transport_benchmark_suite as transport_summary

DOCS_PRESET = "docs-core-runtime"
MAXIMUM_PUBLISHABLE_CV = 0.1
LATENCY_BEGIN = "{/* BENCHMARK-SUITE:GRAPH-LATENCY:BEGIN */}"
LATENCY_END = "{/* BENCHMARK-SUITE:GRAPH-LATENCY:END */}"
THROUGHPUT_BEGIN = "{/* BENCHMARK-SUITE:GRAPH-THROUGHPUT:BEGIN */}"
THROUGHPUT_END = "{/* BENCHMARK-SUITE:GRAPH-THROUGHPUT:END */}"
PROVENANCE_BEGIN = "{/* BENCHMARK-SUITE:PROVENANCE:BEGIN */}"
PROVENANCE_END = "{/* BENCHMARK-SUITE:PROVENANCE:END */}"
PROVENANCE_CASE_ID = "graph_intraprocess_docs_pod_1to4_per_sink_throughput"


@dataclass(frozen=True)
class RowSpec:
    case_stem: str
    topology: str
    payload: str
    storage: str
    notes: str


INTRAPROCESS_ROWS = (
    RowSpec("pod_1to1", "Single-process 1:1", "`BenchmarkPod64`", "Inline", "Sleeping wait set"),
    RowSpec(
        "schema_1to1",
        "Single-process 1:1",
        "`schema::TimeT`",
        "Inline",
        "Generated-schema encode and decode",
    ),
    RowSpec(
        "tensor_pinned_host_1to1",
        "Single-process 1:1",
        "`Tensor`",
        "Pinned host",
        "Descriptor-only; backing pools initialized during warmup",
    ),
    RowSpec(
        "tensor_device_1to1",
        "Single-process 1:1",
        "`Tensor`",
        "Device",
        "Descriptor-only; backing pools initialized during warmup",
    ),
    RowSpec(
        "multi_tensor_pageable_host_1to1",
        "Single-process 1:1",
        "`runtime_benchmark::MultiTensorMessageT`",
        "Pageable host",
        "Two rank-1 uint8 Tensor fields; independent preinitialized pools",
    ),
    RowSpec(
        "image_pageable_host_1to1",
        "Single-process 1:1",
        "`schema::ImageT`",
        "Pageable host",
        "Generated schema with preinitialized Tensor backing",
    ),
    RowSpec(
        "pod_1to2",
        "Single-process 1:2, shared partition",
        "`BenchmarkPod64`",
        "Inline",
        "Shared operator partition; HoloMQ fanout",
    ),
    RowSpec(
        "pod_1to2_per_sink",
        "Single-process 1:2, separate partitions",
        "`BenchmarkPod64`",
        "Inline",
        "One source partition and one partition per subscriber; HoloMQ fanout",
    ),
    RowSpec(
        "pod_1to4",
        "Single-process 1:4, shared partition",
        "`BenchmarkPod64`",
        "Inline",
        "Shared operator partition; HoloMQ fanout",
    ),
    RowSpec(
        "pod_1to4_per_sink",
        "Single-process 1:4, separate partitions",
        "`BenchmarkPod64`",
        "Inline",
        "One source partition and one partition per subscriber; HoloMQ fanout",
    ),
    RowSpec(
        "chain_10_operators",
        "Single-process 10-operator chain",
        "`BenchmarkPod64`",
        "Inline",
        "Source, eight relays, and sink",
    ),
)

INTERPROCESS_ROWS = (
    RowSpec("pod_1to1", "Two-process 1:1", "`BenchmarkPod64`", "Inline", "Sleeping wait set"),
    RowSpec(
        "schema_1to1",
        "Two-process 1:1",
        "`schema::TimeT`",
        "Inline",
        "Generated-schema encode and decode",
    ),
    RowSpec(
        "tensor_pinned_host_1to1",
        "Two-process 1:1",
        "`Tensor`",
        "Pinned host",
        "Preinitialized backing",
    ),
    RowSpec(
        "tensor_device_1to1",
        "Two-process 1:1",
        "`Tensor`",
        "Device",
        "Preinitialized backing",
    ),
    RowSpec(
        "image_pinned_host_1to1",
        "Two-process 1:1",
        "`schema::ImageT`",
        "Pinned host",
        "Generated schema with preinitialized Tensor backing",
    ),
    RowSpec(
        "image_device_1to1",
        "Two-process 1:1",
        "`schema::ImageT`",
        "Device",
        "Generated schema with preinitialized Tensor backing",
    ),
    RowSpec(
        "multi_tensor_loan_pageable_host_1to1",
        "Two-process 1:1",
        "`runtime_benchmark::MultiTensorMessageT`",
        "Pageable host",
        "Two rank-1 uint8 Tensor fields; independent preinitialized pools",
    ),
    RowSpec("pod_1to2", "Three-process 1:2", "`BenchmarkPod64`", "Inline", "Fanout"),
    RowSpec(
        "chain_3_operators",
        "Three-process chain",
        "`BenchmarkPod64`",
        "Inline",
        "Producer, relay, and consumer",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output Markdown path (default: SUITE_DIR/core-runtime-tables.md)",
    )
    parser.add_argument(
        "--update-doc",
        type=Path,
        help="Also replace the generated table regions in benchmarks_core_runtime.mdx",
    )
    return parser.parse_args()


def case_id(domain: str, spec: RowSpec, workload: str) -> str:
    return f"graph_{domain}_docs_{spec.case_stem}_{workload}"


def expected_case_ids() -> set[str]:
    result: set[str] = set()
    for domain, specs in (
        ("intraprocess", INTRAPROCESS_ROWS),
        ("interprocess", INTERPROCESS_ROWS),
    ):
        for spec in specs:
            for workload in ("latency", "throughput"):
                result.add(case_id(domain, spec, workload))
    return result


def manifest_case_id(row: dict[str, Any]) -> str:
    native_runs = row.get("native_runs")
    if not isinstance(native_runs, list) or len(native_runs) != 1:
        raise RuntimeError("a docs Runtime row must map to exactly one suite case")
    value = native_runs[0].get("manifest_case_id")
    if not isinstance(value, str):
        raise RuntimeError("a docs Runtime row is missing its manifest case identifier")
    return value


def require_complete_docs_run(
    manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    if manifest.get("preset") != DOCS_PRESET:
        raise RuntimeError(f"expected --preset {DOCS_PRESET}, found {manifest.get('preset')!r}")
    if manifest.get("repetitions") != 7:
        raise RuntimeError("docs Runtime tables require the preset's seven production repetitions")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("suite manifest is missing cases")
    case_ids = [str(case.get("case_id")) for case in cases if isinstance(case, dict)]
    duplicate_case_ids = sorted(value for value, count in Counter(case_ids).items() if count > 1)
    if duplicate_case_ids:
        raise RuntimeError(
            "suite manifest contains duplicate case identifiers: " + ", ".join(duplicate_case_ids)
        )
    actual_ids = set(case_ids)
    expected_ids = expected_case_ids()
    if actual_ids != expected_ids:
        raise RuntimeError("docs Runtime suite case inventory does not match the table contract")
    incomplete = [
        str(case.get("case_id"))
        for case in cases
        if isinstance(case, dict) and case.get("status") != "passed"
    ]
    if incomplete:
        raise RuntimeError("docs Runtime suite has incomplete cases: " + ", ".join(incomplete))
    row_ids = [manifest_case_id(row) for row in rows]
    duplicate_row_ids = sorted(value for value, count in Counter(row_ids).items() if count > 1)
    if duplicate_row_ids:
        raise RuntimeError(
            "normalized Runtime rows contain duplicate case identifiers: "
            + ", ".join(duplicate_row_ids)
        )
    by_id = dict(zip(row_ids, rows, strict=True))
    if set(by_id) != expected_ids:
        raise RuntimeError("normalized Runtime rows do not match the table contract")
    return by_id


def format_duration_ns(value: Any) -> str:
    numeric = transport_summary.number(value)
    if numeric is None:
        raise RuntimeError("production docs latency row is missing a requested percentile")
    return f"{numeric / 1_000.0:,.3f} µs"


def format_rate(value: Any) -> str:
    numeric = transport_summary.number(value)
    if numeric is None:
        raise RuntimeError("production docs throughput row is missing a requested rate")
    return f"{numeric:,.0f}"


def format_bytes(value: Any) -> str:
    byte_count = int(value)
    if byte_count > 0 and byte_count % (1 << 20) == 0:
        return f"{byte_count // (1 << 20)} MiB"
    if byte_count > 0 and byte_count % (1 << 10) == 0:
        return f"{byte_count // (1 << 10)} KiB"
    return f"{byte_count} B"


def row_notes(domain: str, spec: RowSpec, row: dict[str, Any]) -> str:
    notes = spec.notes
    if domain != "intraprocess" or not spec.case_stem.startswith(("pod_1to2", "pod_1to4")):
        return notes
    event_wake = row.get("event_wake")
    support_threads = str(row.get("runtime_support_threads", ""))
    if isinstance(event_wake, bool):
        notes += "; event wake" if event_wake else "; eventless"
    if support_threads:
        notes += "; channel pump" if "channel_pump" in support_threads.split(",") else "; pump-free"
    return notes


def render_latency_table(
    title: str,
    domain: str,
    specs: tuple[RowSpec, ...],
    by_id: dict[str, dict[str, Any]],
    *,
    include_p999: bool,
) -> str:
    percentile_header = "p50 | p99 | p99.9" if include_p999 else "p50 | p99"
    percentile_rule = "---: | ---: | ---:" if include_p999 else "---: | ---:"
    lines = [
        f"### {title}",
        "",
        f"| Topology | Payload | Storage | Application payload | {percentile_header} | Notes |",
        f"| --- | --- | --- | ---: | {percentile_rule} | --- |",
    ]
    for spec in specs:
        row = by_id[case_id(domain, spec, "latency")]
        percentiles = [
            format_duration_ns(row["latency_p50_ns"]),
            format_duration_ns(row["latency_p99_ns"]),
        ]
        if include_p999:
            percentiles.append(format_duration_ns(row["latency_p999_ns"]))
        lines.append(
            "| "
            + " | ".join(
                [
                    spec.topology,
                    spec.payload,
                    spec.storage,
                    format_bytes(row["application_payload_bytes"]),
                    *percentiles,
                    row_notes(domain, spec, row),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_throughput_table(
    title: str,
    domain: str,
    specs: tuple[RowSpec, ...],
    by_id: dict[str, dict[str, Any]],
) -> str:
    lines = [
        f"### {title}",
        "",
        (
            "| Topology | Payload | Storage | Application payload | "
            "Logical messages/s | Aggregate deliveries/s | Notes |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for spec in specs:
        row = by_id[case_id(domain, spec, "throughput")]
        notes = row_notes(domain, spec, row)
        coefficient_of_variation = transport_summary.number(
            row.get("primary_metric_coefficient_of_variation")
        )
        if (
            coefficient_of_variation is not None
            and coefficient_of_variation > MAXIMUM_PUBLISHABLE_CV
        ):
            notes += (
                f"; run-to-run CV {coefficient_of_variation:.1%} exceeds "
                f"the {MAXIMUM_PUBLISHABLE_CV:.0%} publication gate"
            )
        lines.append(
            "| "
            + " | ".join(
                (
                    spec.topology,
                    spec.payload,
                    spec.storage,
                    format_bytes(row["application_payload_bytes"]),
                    format_rate(row["logical_messages_per_second"]),
                    format_rate(row["aggregate_deliveries_per_second"]),
                    notes,
                )
            )
            + " |"
        )
    return "\n".join(lines)


def render_tables(by_id: dict[str, dict[str, Any]]) -> tuple[str, str]:
    latency = "\n\n".join(
        (
            render_latency_table(
                "Single-process Graph transport latency",
                "intraprocess",
                INTRAPROCESS_ROWS,
                by_id,
                include_p999=True,
            ),
            render_latency_table(
                "Interprocess Graph transport latency",
                "interprocess",
                INTERPROCESS_ROWS,
                by_id,
                include_p999=False,
            ),
        )
    )
    throughput = "\n\n".join(
        (
            render_throughput_table(
                "Single-process Graph throughput",
                "intraprocess",
                INTRAPROCESS_ROWS,
                by_id,
            ),
            render_throughput_table(
                "Interprocess Graph throughput",
                "interprocess",
                INTERPROCESS_ROWS,
                by_id,
            ),
        )
    )
    return latency, throughput


def artifact_manifest(suite_dir: Path, by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = by_id[PROVENANCE_CASE_ID]
    native_run = row["native_runs"][0]
    artifact = native_run.get("artifact")
    if not isinstance(artifact, str):
        raise RuntimeError("provenance row is missing its native artifact")
    artifact_path = (suite_dir / artifact).resolve()
    try:
        artifact_path.relative_to(suite_dir)
    except ValueError as error:
        raise RuntimeError("provenance artifact escapes the suite directory") from error
    manifest_path = artifact_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("--update-doc requires provenance from the native artifact manifest")
    return transport_summary.read_json(manifest_path)


def render_provenance(
    suite_dir: Path, manifest: dict[str, Any], by_id: dict[str, dict[str, Any]]
) -> str:
    summary_path = suite_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("--update-doc requires the suite summary.json artifact")
    artifact_digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    created = transport_summary.number(manifest.get("created_unix_seconds"))
    if created is None:
        raise RuntimeError("suite manifest is missing its creation timestamp")
    run_date = datetime.fromtimestamp(created, tz=timezone.utc).date().isoformat()

    native_manifest = artifact_manifest(suite_dir, by_id)
    machine = native_manifest.get("machine")
    kernel = native_manifest.get("kernel")
    cpu_topology = native_manifest.get("cpu_topology")
    selected_cpus = manifest.get("cpus")
    build = native_manifest.get("build")
    gpus = native_manifest.get("gpus")
    if (
        not isinstance(machine, str)
        or not isinstance(kernel, str)
        or not isinstance(cpu_topology, list)
        or not isinstance(selected_cpus, list)
        or not selected_cpus
        or not all(isinstance(cpu, int) and not isinstance(cpu, bool) for cpu in selected_cpus)
        or not isinstance(build, dict)
        or not isinstance(gpus, list)
        or not gpus
    ):
        raise RuntimeError("native artifact manifest has incomplete hardware provenance")
    topology_by_cpu = {
        item.get("cpu"): item
        for item in cpu_topology
        if isinstance(item, dict) and item.get("pinned") is True
    }
    if set(topology_by_cpu) != set(selected_cpus):
        raise RuntimeError("native artifact does not cover every selected CPU")
    physical_cores = {
        (item.get("package_id"), item.get("core_id")) for item in topology_by_cpu.values()
    }
    packages = {item.get("package_id") for item in topology_by_cpu.values()}
    if any(value is None for core in physical_cores for value in core) or None in packages:
        raise RuntimeError("native artifact is missing physical CPU topology")
    governors = {str(item.get("governor")) for item in topology_by_cpu.values()}
    smt_reported = any(
        "," in str(item.get("thread_siblings", "")) for item in topology_by_cpu.values()
    )

    gpu_values = []
    for gpu in gpus:
        if not isinstance(gpu, dict) or not isinstance(gpu.get("name"), str):
            raise RuntimeError("native artifact has invalid GPU provenance")
        driver = gpu.get("driver_version")
        suffix = f" (driver {driver})" if isinstance(driver, str) else ""
        gpu_values.append(f"{gpu['name']}{suffix}")
    build_type = build.get("cmake_build_type")
    cxx_version = build.get("cxx_compiler_version")
    cuda_version = build.get("cuda_compiler_version_output")
    if not all(isinstance(value, str) for value in (build_type, cxx_version, cuda_version)):
        raise RuntimeError("native artifact is missing build provenance")

    coefficients = [
        transport_summary.number(row.get("primary_metric_coefficient_of_variation"))
        for row in by_id.values()
    ]
    if any(value is None for value in coefficients):
        raise RuntimeError("docs Runtime rows are missing variability evidence")
    passing_rows = sum(
        value <= MAXIMUM_PUBLISHABLE_CV for value in coefficients if value is not None
    )
    validation_modes = {str(row.get("payload_validation")) for row in by_id.values()}
    if validation_modes != {"none"}:
        raise RuntimeError("docs Runtime rows do not share the production validation policy")

    def latency_policy(domain: str) -> tuple[int, int]:
        rows = [
            by_id[case_id(domain, spec, "latency")]
            for spec in (INTRAPROCESS_ROWS if domain == "intraprocess" else INTERPROCESS_ROWS)
        ]
        samples = [
            transport_summary.optional_nonnegative_int(row.get("measured_samples")) for row in rows
        ]
        warmups = [
            transport_summary.optional_nonnegative_int(row.get("warmup_samples")) for row in rows
        ]
        if any(value is None for value in (*samples, *warmups)):
            raise RuntimeError("docs Runtime latency rows are missing sample-count provenance")
        return min(value for value in samples if value is not None), min(
            value for value in warmups if value is not None
        )

    intraprocess_samples, intraprocess_warmup = latency_policy("intraprocess")
    interprocess_samples, interprocess_warmup = latency_policy("interprocess")
    controls = ", ".join(sorted(governors)) + " governor reported for selected CPUs"
    controls += "; ASLR and CPU idle-state status are not recorded in suite artifacts"
    cpu_description = (
        f"{machine}; {len(physical_cores)} selected physical cores on "
        f"{len(packages)} socket{'s' if len(packages) != 1 else ''}"
    )
    if smt_reported:
        cpu_description += "; SMT sibling topology reported"
    rows = (
        ("Run date and artifact digest", f"{run_date}; `summary.json` SHA-256 `{artifact_digest}`"),
        ("CPU and topology", cpu_description),
        ("Selected CPUs", "Logical CPUs " + ", ".join(str(cpu) for cpu in selected_cpus)),
        ("GPU", "; ".join(gpu_values)),
        ("OS and kernel", f"Linux {kernel} on {machine}"),
        ("Build", f"{build_type}; {cxx_version}; {cuda_version}"),
        ("System controls", controls),
        (
            "Preset and repetitions",
            f"`{manifest['preset']}`; {manifest['repetitions']} repetitions",
        ),
        (
            "Run-to-run gate",
            f"{MAXIMUM_PUBLISHABLE_CV:.0%} maximum primary-metric coefficient of variation; "
            f"{passing_rows}/{len(by_id)} rows passed",
        ),
        (
            "Single-process latency",
            f"At least {intraprocess_samples:,} measured samples; {intraprocess_warmup:,} warmup; "
            "p50, p99, and p99.9",
        ),
        (
            "Interprocess latency",
            f"{interprocess_samples:,} measured samples; "
            f"{interprocess_warmup:,} warmup; p50 and p99",
        ),
        (
            "Payload validation",
            "Disabled in timed production rows; structural and completion checks retained",
        ),
    )
    lines = ["| Field | Value |", "| --- | --- |"]
    for field, value in rows:
        escaped_value = value.replace("|", "\\|")
        lines.append(f"| {field} | {escaped_value} |")
    return "\n".join(lines)


def replace_region(text: str, begin: str, end: str, replacement: str) -> str:
    if text.count(begin) != 1 or text.count(end) != 1:
        raise RuntimeError(f"document must contain exactly one {begin} / {end} marker pair")
    prefix, remainder = text.split(begin, 1)
    _, suffix = remainder.split(end, 1)
    return f"{prefix}{begin}\n\n{replacement}\n\n{end}{suffix}"


def update_doc(path: Path, latency: str, throughput: str, provenance: str) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_region(text, LATENCY_BEGIN, LATENCY_END, latency)
    text = replace_region(text, THROUGHPUT_BEGIN, THROUGHPUT_END, throughput)
    text = replace_region(text, PROVENANCE_BEGIN, PROVENANCE_END, provenance)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    manifest, rows = transport_summary.collect_rows(suite_dir)
    by_id = require_complete_docs_run(manifest, rows)
    latency, throughput = render_tables(by_id)
    output = (args.output or (suite_dir / "core-runtime-tables.md")).resolve()
    output.write_text(f"{latency}\n\n{throughput}\n", encoding="utf-8")
    if args.update_doc is not None:
        provenance = render_provenance(suite_dir, manifest, by_id)
        update_doc(args.update_doc.resolve(), latency, throughput, provenance)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
