#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run interleaved production-Graph partition-cover comparisons."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "holoscan.ea2.partition-cover-comparison/v1"
COVERS = ("legacy", "automatic", "include_all", "hybrid")
PROVENANCE_FIELDS = (
    "git_revision",
    "binary",
    "binary_sha256",
    "source_state",
    "source_snapshot_sha256",
    "hostname",
    "kernel",
    "machine",
    "build",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_binary", type=Path)
    parser.add_argument(
        "--runner", type=Path, default=Path(__file__).with_name("run_runtime_graph_benchmark.py")
    )
    parser.add_argument("--baseline-cover", choices=COVERS, default="legacy")
    parser.add_argument("--candidate-cover", choices=COVERS, default="automatic")
    parser.add_argument("--baseline-operator-lanes", choices=("separate", "shared", "per_sink"))
    parser.add_argument("--candidate-operator-lanes", choices=("separate", "shared", "per_sink"))
    parser.add_argument(
        "--workload", choices=("latency", "notification_latency", "saturated"), default="latency"
    )
    parser.add_argument("--scenario", default="pod")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--fanout", type=int, default=1)
    parser.add_argument("--relay-count", type=int, default=1)
    parser.add_argument("--publisher-period-us", type=int, default=1_000)
    parser.add_argument("--controller-cpu", type=int, default=-1)
    parser.add_argument("--max-median-parity", type=float, default=0.03)
    parser.add_argument("--max-median-cv", type=float, default=0.10)
    parser.add_argument(
        "--arm-timeout-sec",
        type=float,
        default=1_800.0,
        help="maximum wall time for each one-repetition runner process",
    )
    parser.add_argument("--git-revision")
    parser.add_argument("--source-state", default="unknown")
    parser.add_argument("--source-snapshot-sha256")
    parser.add_argument("--runtime-phase-counters", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def operator_lanes(cover: str, requested: str | None) -> str:
    if requested is not None:
        return requested
    return "shared" if cover == "include_all" else "separate"


def arm_command(
    args: argparse.Namespace,
    cover: str,
    lanes: str,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(args.runner.resolve()),
        str(args.benchmark_binary.resolve()),
        "--partition-cover",
        cover,
        "--operator-lanes",
        lanes,
        "--workload",
        args.workload,
        "--scenario",
        args.scenario,
        "--samples",
        str(args.samples),
        "--warmup",
        str(args.warmup),
        "--repetitions",
        "1",
        "--fanout",
        str(args.fanout),
        "--relay-count",
        str(args.relay_count),
        "--publisher-period-us",
        str(args.publisher_period_us),
        "--controller-cpu",
        str(args.controller_cpu),
        "--output-dir",
        str(output_dir),
    ]
    if args.git_revision:
        command.extend(("--git-revision", args.git_revision))
    command.extend(("--source-state", args.source_state))
    if args.source_snapshot_sha256:
        command.extend(("--source-snapshot-sha256", args.source_snapshot_sha256))
    if args.runtime_phase_counters:
        command.append("--runtime-phase-counters")
    return command


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    return {
        "samples": len(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "coefficient_of_variation": statistics.pstdev(values) / mean if mean else 0.0,
    }


def metric_key(workload: str, scenario: str) -> str:
    if workload == "saturated":
        return (
            "source_activations_per_second"
            if scenario == "source_only"
            else "logical_publications_per_second"
        )
    return "latency_p50_ns"


def metric_value(summary: dict[str, Any], workload: str, scenario: str) -> float:
    run = summary["runs"][0]
    key = metric_key(workload, scenario)
    value = run.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"runtime summary is missing numeric {key}")
    return float(value)


def relative_regression(baseline: float, candidate: float, workload: str) -> float:
    if baseline == 0.0:
        raise RuntimeError("runtime comparison baseline metric is zero")
    return (
        (candidate - baseline) / baseline
        if workload != "saturated"
        else (baseline - candidate) / baseline
    )


def concurrency_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    """Retain representative correctness, overlap, and owner evidence from one arm run."""
    run = summary["runs"][0]
    keys = (
        "counts",
        "latency_p50_ns",
        "logical_publications_per_second",
        "source_post_emit_to_sink_overlaps",
        "source_post_emit_to_sink_by_sink_overlaps",
        "source_pre_return_to_sink_callback_by_sink_overlaps",
        "source_to_sink_by_sink_p50_ns",
    )
    return {
        "validation": summary.get("validation"),
        "callback_thread_evidence": summary.get("callback_thread_evidence"),
        **{key: run.get(key) for key in keys if key in run},
    }


def fanout_concurrency_proof(
    args: argparse.Namespace,
    covers: dict[str, str],
    arm_summaries: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    applicable = (
        args.workload == "notification_latency"
        and args.fanout > 1
        and set(covers.values()) == {"automatic", "include_all"}
    )
    if not applicable:
        return {"applicable": False, "passed": None}
    automatic_arm = next(arm for arm, cover in covers.items() if cover == "automatic")
    include_all_arm = next(arm for arm, cover in covers.items() if cover == "include_all")
    automatic_evidence = [concurrency_evidence(row) for row in arm_summaries[automatic_arm]]
    include_all_evidence = [concurrency_evidence(row) for row in arm_summaries[include_all_arm]]

    def causal(evidence: dict[str, Any]) -> bool:
        validation = evidence.get("validation")
        counts = evidence.get("counts")
        return (
            isinstance(validation, dict)
            and validation.get("passed") is True
            and isinstance(counts, dict)
            and counts.get("validation_errors") == 0
            and isinstance(counts.get("per_sink"), list)
            and len(counts["per_sink"]) == args.fanout
            and all(
                isinstance(item, dict)
                and isinstance(item.get("deliveries"), int)
                and item["deliveries"] > 0
                and isinstance(item.get("final_sequence"), int)
                and item["final_sequence"] > 0
                for item in counts["per_sink"]
            )
        )

    def automatic_owners(evidence: dict[str, Any]) -> bool:
        callback = evidence.get("callback_thread_evidence")
        if not isinstance(callback, dict) or callback.get("valid") is not True:
            return False
        roles = callback.get("roles")
        sinks = callback.get("sink_roles")
        if not isinstance(roles, list) or not isinstance(sinks, list) or len(sinks) != args.fanout:
            return False
        source = next((role for role in roles if role.get("role") == "source"), None)
        sink_tids = [sink.get("native_tid") for sink in sinks]
        return (
            isinstance(source, dict)
            and isinstance(source.get("native_tid"), int)
            and all(
                isinstance(tid, int) and sink.get("multiple_threads") is False
                for tid, sink in zip(sink_tids, sinks, strict=True)
            )
            and len(set(sink_tids)) == args.fanout
            and source["native_tid"] not in sink_tids
        )

    def include_all_owner(evidence: dict[str, Any]) -> bool:
        callback = evidence.get("callback_thread_evidence")
        return (
            isinstance(callback, dict)
            and callback.get("valid") is True
            and callback.get("shared_lane_single_thread") is True
            and callback.get("callbacks_on_partition_owner") is True
        )

    overlap_count = sum(
        sum(evidence.get("source_pre_return_to_sink_callback_by_sink_overlaps", []))
        for evidence in automatic_evidence
    )
    distinct_partition_owners = all(automatic_owners(row) for row in automatic_evidence)
    include_all_single_owner = all(include_all_owner(row) for row in include_all_evidence)
    causal_correctness = all(causal(row) for row in automatic_evidence + include_all_evidence)
    return {
        "applicable": True,
        "passed": (
            distinct_partition_owners
            and include_all_single_owner
            and causal_correctness
            and overlap_count > 0
        ),
        "automatic_arm": automatic_arm,
        "include_all_arm": include_all_arm,
        "distinct_partition_owners": distinct_partition_owners,
        "include_all_single_owner": include_all_single_owner,
        "causal_correctness": causal_correctness,
        "source_pre_return_to_sink_callback_overlap_count": overlap_count,
    }


def owner_geometry(summary: dict[str, Any]) -> dict[str, Any]:
    placement = summary["placement"]
    uses = placement["operator_partition_uses"]
    support_affinity = [
        {
            "role": item.get("role"),
            "requested_cpu": item.get("requested_cpu"),
            "observed_affinity": item.get("observed_affinity"),
            "entered": item.get("entered"),
        }
        for item in placement.get("support_threads", [])
        if item.get("entered") is True
    ]
    return {
        "partition_count": placement["operator_lane_count"],
        "operator_counts": sorted(item["operator_count"] for item in uses),
        "ownership": (
            "controller_inline"
            if placement["operator_lane_count"] == 1
            else "execution_lane_workers"
        ),
        "support_affinity": support_affinity,
    }


def cpu_topology(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest.get("cpu_topology"), sort_keys=True)


def valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def comparison_provenance(manifests: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Retain arm-specific identity and claim shared provenance only when every run agrees."""
    arms = {
        arm: {key: manifests[arm][0].get(key) for key in PROVENANCE_FIELDS}
        for arm in ("baseline", "candidate")
    }
    result: dict[str, Any] = {"arms": arms}
    records = [
        {key: manifest.get(key) for key in PROVENANCE_FIELDS}
        for arm_manifests in manifests.values()
        for manifest in arm_manifests
    ]
    if records and all(record == records[0] for record in records[1:]):
        result["shared"] = records[0]
    return result


def classify(
    args: argparse.Namespace,
    arm_summaries: dict[str, list[dict[str, Any]]],
    manifests: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    issues: list[str] = []
    if args.repetitions < 5:
        issues.append("insufficient_repetitions")
    required_arms = {"baseline", "candidate"}
    if set(arm_summaries) != required_arms or any(
        len(arm_summaries.get(arm, [])) != args.repetitions for arm in required_arms
    ):
        issues.append("summary_inventory_incomplete")
    if set(manifests) != required_arms or any(
        len(manifests.get(arm, [])) != args.repetitions for arm in required_arms
    ):
        issues.append("manifest_inventory_incomplete")
    signatures = {
        arm: {summary["partitioning"]["cover_signature"] for summary in summaries}
        for arm, summaries in arm_summaries.items()
    }
    if any(len(values) != 1 for values in signatures.values()):
        issues.append("cover_signature_drift")
    elif signatures["baseline"] != signatures["candidate"]:
        issues.append("partition_covers_not_equivalent")
    geometries = {
        arm: {json.dumps(owner_geometry(summary), sort_keys=True) for summary in summaries}
        for arm, summaries in arm_summaries.items()
    }
    if any(len(values) != 1 for values in geometries.values()):
        issues.append("partition_owner_geometry_drift")
    elif geometries["baseline"] != geometries["candidate"]:
        issues.append("partition_owner_geometry_not_equivalent")
    for summaries in arm_summaries.values():
        for summary in summaries:
            geometry = owner_geometry(summary)
            if geometry["ownership"] == "execution_lane_workers":
                issues.append("operator_owner_actual_affinity_unavailable")
            for item in geometry["support_affinity"]:
                requested_cpu = item["requested_cpu"]
                observed_affinity = item["observed_affinity"]
                if (
                    not isinstance(requested_cpu, int)
                    or not isinstance(observed_affinity, list)
                    or observed_affinity != [requested_cpu]
                ):
                    issues.append("support_thread_actual_affinity_not_confirmed")
    all_manifests = [manifest for arm in manifests.values() for manifest in arm]
    for manifest in all_manifests:
        topology = manifest.get("cpu_topology")
        if not isinstance(topology, list) or any(not item.get("pinned") for item in topology):
            issues.append("runtime_thread_cpu_affinity_unpinned")
            break
        if any(item.get("governor") != "performance" for item in topology):
            issues.append("operator_cpu_governor_not_performance")
            break
    topologies = {
        arm: {cpu_topology(manifest) for manifest in arm_manifests}
        for arm, arm_manifests in manifests.items()
    }
    if any(len(values) != 1 for values in topologies.values()):
        issues.append("cpu_topology_or_affinity_drift")
    elif topologies["baseline"] != topologies["candidate"]:
        issues.append("cpu_topology_or_affinity_not_equivalent")
    host_fields = ("hostname", "kernel", "machine")
    if any(
        not all(isinstance(manifest.get(field), str) and manifest[field] for field in host_fields)
        for manifest in all_manifests
    ):
        issues.append("host_identity_missing")
    host_identities = {
        tuple(manifest.get(field) for field in host_fields) for manifest in all_manifests
    }
    if len(host_identities) != 1:
        issues.append("host_identity_drift")
    revisions = {manifest.get("git_revision") for manifest in all_manifests}
    builds = {json.dumps(manifest.get("build"), sort_keys=True) for manifest in all_manifests}
    if any(
        not isinstance(manifest.get("git_revision"), str) or not manifest["git_revision"]
        for manifest in all_manifests
    ):
        issues.append("git_revision_missing")
    if any(
        not isinstance(manifest.get("build"), dict) or not manifest["build"]
        for manifest in all_manifests
    ):
        issues.append("build_fingerprint_missing")
    if len(revisions) != 1:
        issues.append("git_revision_drift")
    if len(builds) != 1:
        issues.append("build_fingerprint_drift")
    binary_hashes = {manifest.get("binary_sha256") for manifest in all_manifests}
    if any(not valid_sha256(manifest.get("binary_sha256")) for manifest in all_manifests):
        issues.append("binary_sha256_missing")
    if len(binary_hashes) != 1:
        issues.append("binary_sha256_drift")
    source_states = {manifest.get("source_state") for manifest in all_manifests}
    if any(
        not isinstance(manifest.get("source_state"), str)
        or not manifest["source_state"]
        or manifest["source_state"] == "unknown"
        for manifest in all_manifests
    ):
        issues.append("source_state_missing")
    if len(source_states) != 1:
        issues.append("source_state_drift")
    source_hashes = {manifest.get("source_snapshot_sha256") for manifest in all_manifests}
    if any(not valid_sha256(manifest.get("source_snapshot_sha256")) for manifest in all_manifests):
        issues.append("source_snapshot_sha256_missing")
    if len(source_hashes) != 1:
        issues.append("source_snapshot_sha256_drift")
    return {"controlled": not issues, "issues": sorted(set(issues))}


def main() -> int:
    args = parse_args()
    if args.repetitions <= 0 or args.samples <= 0 or args.warmup < 0 or args.arm_timeout_sec <= 0:
        raise ValueError("repetitions and samples must be positive and warmup nonnegative")
    if not 0.0 <= args.max_median_parity <= 1.0 or args.max_median_cv < 0.0:
        raise ValueError("comparison thresholds must be nonnegative fractions")
    if not args.source_state:
        raise ValueError("--source-state must be nonempty")
    if args.source_snapshot_sha256 is not None and not valid_sha256(args.source_snapshot_sha256):
        raise ValueError("--source-snapshot-sha256 must be a lowercase SHA-256 digest")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lanes = {
        "baseline": operator_lanes(args.baseline_cover, args.baseline_operator_lanes),
        "candidate": operator_lanes(args.candidate_cover, args.candidate_operator_lanes),
    }
    covers = {"baseline": args.baseline_cover, "candidate": args.candidate_cover}
    summaries: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    manifests: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    commands: list[dict[str, Any]] = []
    order: list[str] = []
    for repetition in range(args.repetitions):
        pair_order = ("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")
        for arm in pair_order:
            order.append(arm)
            run_dir = output_dir / f"pair_{repetition:03d}_{arm}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = arm_command(args, covers[arm], lanes[arm], run_dir)
            (run_dir / "comparison_runner_command.json").write_text(
                json.dumps(command, indent=2) + "\n", encoding="utf-8"
            )
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=args.arm_timeout_sec,
                )
            except subprocess.TimeoutExpired as error:
                stdout = (
                    error.stdout.decode(errors="replace")
                    if isinstance(error.stdout, bytes)
                    else error.stdout
                )
                stderr = (
                    error.stderr.decode(errors="replace")
                    if isinstance(error.stderr, bytes)
                    else error.stderr
                )
                (run_dir / "comparison_runner_stdout.log").write_text(
                    stdout or "", encoding="utf-8"
                )
                (run_dir / "comparison_runner_stderr.log").write_text(
                    stderr or "", encoding="utf-8"
                )
                raise RuntimeError(
                    f"{arm} repetition {repetition} timed out after "
                    f"{args.arm_timeout_sec} seconds: {run_dir}"
                ) from error
            (run_dir / "comparison_runner_stdout.log").write_text(
                completed.stdout, encoding="utf-8"
            )
            (run_dir / "comparison_runner_stderr.log").write_text(
                completed.stderr, encoding="utf-8"
            )
            commands.append({"arm": arm, "command": command, "returncode": completed.returncode})
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{arm} repetition {repetition} failed with {completed.returncode}"
                )
            summaries[arm].append(
                json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            )
            manifests[arm].append(
                json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            )
    values = {
        arm: [metric_value(summary, args.workload, args.scenario) for summary in arm_summaries]
        for arm, arm_summaries in summaries.items()
    }
    statistics_by_arm = {arm: summarize(arm_values) for arm, arm_values in values.items()}
    qualification = classify(args, summaries, manifests)
    concurrency_proof = fanout_concurrency_proof(args, covers, summaries)
    if concurrency_proof["applicable"] and concurrency_proof["passed"] is not True:
        qualification["issues"].append("fanout_concurrency_evidence_incomplete")
    for arm, arm_statistics in statistics_by_arm.items():
        if arm_statistics["coefficient_of_variation"] > args.max_median_cv:
            qualification["issues"].append(f"{arm}_median_cv_exceeded")
    qualification["issues"] = sorted(set(qualification["issues"]))
    qualification["controlled"] = not qualification["issues"]
    baseline_median = float(statistics_by_arm["baseline"]["median"])
    candidate_median = float(statistics_by_arm["candidate"]["median"])
    observed_relative_regression = relative_regression(
        baseline_median, candidate_median, args.workload
    )
    parity = {
        "threshold": args.max_median_parity,
        "relative_regression": observed_relative_regression,
        "status": (
            "unqualified"
            if not qualification["controlled"]
            else "pass"
            if observed_relative_regression <= args.max_median_parity
            else "fail"
        ),
    }
    result = {
        "schema": SCHEMA,
        "comparison": {"baseline": covers["baseline"], "candidate": covers["candidate"]},
        "workload": args.workload,
        "scenario": args.scenario,
        "interleave_order": order,
        "metric": metric_key(args.workload, args.scenario),
        "arms": {
            arm: {
                "cover": covers[arm],
                "requested_operator_lanes": lanes[arm],
                "statistics": statistics_by_arm[arm],
                "values": values[arm],
                "partitioning": summaries[arm][0]["partitioning"],
                "owner_geometry": owner_geometry(summaries[arm][0]),
                "evidence_runs": [concurrency_evidence(summary) for summary in summaries[arm]],
            }
            for arm in ("baseline", "candidate")
        },
        "qualification": qualification,
        "parity": parity,
        "fanout_concurrency_proof": concurrency_proof,
        "provenance": comparison_provenance(manifests),
        "commands": commands,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if parity["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
