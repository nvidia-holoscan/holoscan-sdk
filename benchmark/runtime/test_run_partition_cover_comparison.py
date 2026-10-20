# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_partition_cover_comparison as comparison


def repetitions(row: dict, count: int = 5) -> list[dict]:
    return [copy.deepcopy(row) for _ in range(count)]


def args(**updates: object) -> SimpleNamespace:
    values = {
        "runner": Path("runner.py"),
        "benchmark_binary": Path("benchmark"),
        "workload": "latency",
        "scenario": "pod",
        "samples": 100,
        "warmup": 10,
        "repetitions": 5,
        "fanout": 1,
        "relay_count": 1,
        "publisher_period_us": 1_000,
        "controller_cpu": -1,
        "arm_timeout_sec": 30.0,
        "git_revision": "abc123",
        "source_state": "dirty-task040-worktree-snapshot",
        "source_snapshot_sha256": "b" * 64,
        "runtime_phase_counters": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def summary(signature: str = "2:6:source;", operator_counts: list[int] | None = None) -> dict:
    counts = operator_counts or [2]
    return {
        "partitioning": {"cover_signature": signature},
        "placement": {
            "operator_lane_count": len(counts),
            "operator_partition_uses": [
                {"operator_count": count, "partition": f"p-{index}", "lane": index}
                for index, count in enumerate(counts)
            ],
            "support_threads": [
                {
                    "role": "controller",
                    "requested_cpu": 2,
                    "observed_affinity": [2],
                    "entered": True,
                }
            ],
        },
        "runs": [{"latency_p50_ns": 100, "logical_publications_per_second": 1_000}],
    }


def manifest(*, pinned: bool = True, hostname: str = "benchmark-host", cpu: int = 2) -> dict:
    return {
        "hostname": hostname,
        "kernel": "6.8.0",
        "machine": "x86_64",
        "cpu_topology": [
            {
                "role": "controller",
                "cpu": cpu,
                "pinned": pinned,
                "governor": "performance",
                "package_id": 0,
                "core_id": cpu,
                "thread_siblings": str(cpu),
                "topology_available": True,
            }
        ],
        "git_revision": "abc123",
        "binary_sha256": "a" * 64,
        "source_state": "dirty-task040-worktree-snapshot",
        "source_snapshot_sha256": "b" * 64,
        "build": {"type": "Release"},
    }


def test_arm_command_forwards_cover_and_single_repetition(tmp_path: Path) -> None:
    command = comparison.arm_command(args(), "automatic", "separate", tmp_path)

    assert command[command.index("--partition-cover") + 1] == "automatic"
    assert command[command.index("--operator-lanes") + 1] == "separate"
    assert command[command.index("--repetitions") + 1] == "1"


def test_arm_command_forwards_runtime_phase_counters(tmp_path: Path) -> None:
    command = comparison.arm_command(
        args(workload="notification_latency", runtime_phase_counters=True),
        "automatic",
        "separate",
        tmp_path,
    )

    assert command[command.index("--workload") + 1] == "notification_latency"
    assert "--runtime-phase-counters" in command


def test_arm_command_forwards_source_snapshot_provenance(tmp_path: Path) -> None:
    command = comparison.arm_command(args(), "automatic", "separate", tmp_path)

    assert command[command.index("--source-state") + 1] == "dirty-task040-worktree-snapshot"
    assert command[command.index("--source-snapshot-sha256") + 1] == "b" * 64


def test_metric_value_treats_notification_latency_as_latency() -> None:
    assert comparison.metric_value(summary(), "notification_latency", "pod") == 100.0


def test_metric_value_treats_source_only_saturated_as_activation_rate() -> None:
    row = summary()
    row["runs"][0].update(
        {
            "logical_publications_per_second": 0.0,
            "source_activations_per_second": 2_000.0,
        }
    )

    assert comparison.metric_value(row, "saturated", "source_only") == 2_000.0
    assert comparison.metric_key("saturated", "source_only") == "source_activations_per_second"
    assert comparison.relative_regression(2_000.0, 2_100.0, "saturated") == -0.05


def test_concurrency_evidence_retains_correctness_overlap_and_owner_data() -> None:
    row = summary()
    row["validation"] = {"passed": True, "mode": "final"}
    row["callback_thread_evidence"] = {"enabled": True, "valid": True}
    row["runs"][0].update(
        {
            "counts": {"source_callbacks": 100, "sink_callbacks": 400},
            "source_post_emit_to_sink_overlaps": 100,
            "source_post_emit_to_sink_by_sink_overlaps": [100, 100, 100, 100],
            "source_to_sink_by_sink_p50_ns": [10, 11, 12, 13],
        }
    )

    evidence = comparison.concurrency_evidence(row)

    assert evidence["validation"]["passed"] is True
    assert evidence["counts"]["sink_callbacks"] == 400
    assert evidence["source_post_emit_to_sink_overlaps"] == 100
    assert evidence["callback_thread_evidence"]["valid"] is True


def test_fanout_concurrency_proof_requires_distinct_tids_overlap_and_causal_correctness() -> None:
    automatic = summary("1:6:sink1;1:6:sink2;1:6:source;", [1, 1, 1])
    include_all = summary("3:6:sink16:sink26:source;", [3])
    for row, tids, inline in ((automatic, [29, 31], False), (include_all, [17, 17], True)):
        row["validation"] = {"passed": True}
        row["runs"][0].update(
            {
                "counts": {
                    "validation_errors": 0,
                    "per_sink": [
                        {"index": 0, "deliveries": 11, "final_sequence": 11},
                        {"index": 1, "deliveries": 11, "final_sequence": 11},
                    ],
                },
                "source_pre_return_to_sink_callback_by_sink_overlaps": [1, 0]
                if not inline
                else [0, 0],
            }
        )
        row["callback_thread_evidence"] = {
            "valid": True,
            "shared_lane_single_thread": inline,
            "callbacks_on_partition_owner": inline,
            "roles": [{"role": "source", "native_tid": 23 if not inline else 17}],
            "sink_roles": [
                {"index": index, "native_tid": tid, "multiple_threads": False}
                for index, tid in enumerate(tids)
            ],
        }
    rows = {"baseline": repetitions(automatic), "candidate": repetitions(include_all)}
    covers = {"baseline": "automatic", "candidate": "include_all"}

    proof = comparison.fanout_concurrency_proof(
        args(fanout=2, workload="notification_latency"), covers, rows
    )

    assert proof["passed"] is True
    assert proof["distinct_partition_owners"] is True
    assert proof["source_pre_return_to_sink_callback_overlap_count"] == 5

    rows["baseline"][0]["callback_thread_evidence"]["sink_roles"][1]["native_tid"] = 29
    proof = comparison.fanout_concurrency_proof(
        args(fanout=2, workload="notification_latency"), covers, rows
    )
    assert proof["passed"] is False


def test_classify_accepts_controlled_same_cover_and_owner_geometry() -> None:
    rows = {"baseline": repetitions(summary()), "candidate": repetitions(summary())}
    manifests = {"baseline": repetitions(manifest()), "candidate": repetitions(manifest())}

    result = comparison.classify(args(), rows, manifests)

    assert result == {"controlled": True, "issues": []}


def test_classify_marks_noisy_and_non_equivalent_comparisons_unqualified() -> None:
    rows = {
        "baseline": repetitions(summary()),
        "candidate": repetitions(summary("1:6:source;1:4:sink;", [1, 1])),
    }

    manifests = {
        "baseline": repetitions(manifest(pinned=False)),
        "candidate": repetitions(manifest(pinned=False)),
    }
    result = comparison.classify(args(), rows, manifests)

    assert result["controlled"] is False
    assert "partition_covers_not_equivalent" in result["issues"]
    assert "partition_owner_geometry_not_equivalent" in result["issues"]
    assert "runtime_thread_cpu_affinity_unpinned" in result["issues"]


def test_classify_fails_closed_on_host_and_cpu_topology_drift() -> None:
    rows = {"baseline": repetitions(summary()), "candidate": repetitions(summary())}
    manifests = {
        "baseline": repetitions(manifest()),
        "candidate": repetitions(manifest(hostname="other-host", cpu=4)),
    }

    result = comparison.classify(args(), rows, manifests)

    assert result["controlled"] is False
    assert "host_identity_drift" in result["issues"]
    assert "cpu_topology_or_affinity_not_equivalent" in result["issues"]


def test_classify_fails_closed_on_missing_provenance_and_actual_affinity_drift() -> None:
    baseline = summary()
    candidate = summary()
    candidate["placement"]["support_threads"][0]["observed_affinity"] = [4]
    rows = {"baseline": repetitions(baseline), "candidate": repetitions(candidate)}
    incomplete = manifest()
    incomplete.pop("hostname")
    incomplete.pop("git_revision")
    incomplete.pop("binary_sha256")
    incomplete.pop("source_state")
    incomplete.pop("source_snapshot_sha256")
    incomplete["build"] = {}
    manifests = {"baseline": repetitions(incomplete), "candidate": repetitions(incomplete)}

    result = comparison.classify(args(), rows, manifests)

    assert result["controlled"] is False
    assert "partition_owner_geometry_not_equivalent" in result["issues"]
    assert "host_identity_missing" in result["issues"]
    assert "git_revision_missing" in result["issues"]
    assert "binary_sha256_missing" in result["issues"]
    assert "source_state_missing" in result["issues"]
    assert "source_snapshot_sha256_missing" in result["issues"]
    assert "build_fingerprint_missing" in result["issues"]
    assert "support_thread_actual_affinity_not_confirmed" in result["issues"]


def test_classify_fails_closed_when_execution_lane_affinity_is_not_observed() -> None:
    worker_summary = summary(operator_counts=[1, 1])
    rows = {
        "baseline": repetitions(worker_summary),
        "candidate": repetitions(worker_summary),
    }
    manifests = {"baseline": repetitions(manifest()), "candidate": repetitions(manifest())}

    result = comparison.classify(args(), rows, manifests)

    assert result["controlled"] is False
    assert "operator_owner_actual_affinity_unavailable" in result["issues"]


def test_classify_fails_closed_on_binary_and_source_snapshot_drift() -> None:
    rows = {"baseline": repetitions(summary()), "candidate": repetitions(summary())}
    baseline = manifest()
    candidate = manifest()
    candidate["binary_sha256"] = "c" * 64
    candidate["source_state"] = "clean"
    candidate["source_snapshot_sha256"] = "d" * 64
    manifests = {"baseline": repetitions(baseline), "candidate": repetitions(candidate)}

    result = comparison.classify(args(), rows, manifests)

    assert result["controlled"] is False
    assert "binary_sha256_drift" in result["issues"]
    assert "source_state_drift" in result["issues"]
    assert "source_snapshot_sha256_drift" in result["issues"]


def test_comparison_provenance_keeps_drifted_arms_separate() -> None:
    baseline = manifest()
    candidate = manifest()
    candidate["git_revision"] = "def456"
    candidate["binary_sha256"] = "c" * 64
    manifests = {
        "baseline": repetitions(baseline),
        "candidate": repetitions(candidate),
    }

    provenance = comparison.comparison_provenance(manifests)

    assert set(provenance["arms"]) == {"baseline", "candidate"}
    assert provenance["arms"]["baseline"]["git_revision"] == "abc123"
    assert provenance["arms"]["candidate"]["git_revision"] == "def456"
    assert "shared" not in provenance


def test_comparison_provenance_marks_only_identical_arms_shared() -> None:
    manifests = {
        "baseline": repetitions(manifest()),
        "candidate": repetitions(manifest()),
    }

    provenance = comparison.comparison_provenance(manifests)

    assert provenance["shared"] == provenance["arms"]["baseline"]
    assert provenance["shared"] == provenance["arms"]["candidate"]


def test_main_creates_child_directory_before_preserving_failed_process_logs(
    tmp_path: Path, monkeypatch
) -> None:
    run_args = args(
        output_dir=tmp_path / "comparison",
        baseline_cover="legacy",
        candidate_cover="automatic",
        baseline_operator_lanes=None,
        candidate_operator_lanes=None,
        max_median_parity=0.03,
        max_median_cv=0.10,
        repetitions=1,
    )
    monkeypatch.setattr(comparison, "parse_args", lambda: run_args)
    monkeypatch.setattr(
        comparison.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["runner"], returncode=7, stdout="child stdout", stderr="child stderr"
        ),
    )

    with pytest.raises(RuntimeError, match="failed with 7"):
        comparison.main()

    run_dir = run_args.output_dir / "pair_000_baseline"
    assert (run_dir / "comparison_runner_stdout.log").read_text() == "child stdout"
    assert (run_dir / "comparison_runner_stderr.log").read_text() == "child stderr"


def test_main_preserves_partial_output_when_child_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_args = args(
        output_dir=tmp_path / "comparison",
        baseline_cover="legacy",
        candidate_cover="automatic",
        baseline_operator_lanes=None,
        candidate_operator_lanes=None,
        max_median_parity=0.03,
        max_median_cv=0.10,
        repetitions=1,
    )
    monkeypatch.setattr(comparison, "parse_args", lambda: run_args)

    def timed_out(*_args: object, **kwargs: object) -> None:
        assert kwargs["timeout"] == 30.0
        raise subprocess.TimeoutExpired(
            cmd=["runner"], timeout=30.0, output="partial stdout", stderr=b"partial stderr"
        )

    monkeypatch.setattr(comparison.subprocess, "run", timed_out)

    with pytest.raises(RuntimeError, match="baseline repetition 0 timed out"):
        comparison.main()

    run_dir = run_args.output_dir / "pair_000_baseline"
    assert (run_dir / "comparison_runner_stdout.log").read_text() == "partial stdout"
    assert (run_dir / "comparison_runner_stderr.log").read_text() == "partial stderr"
    assert json.loads((run_dir / "comparison_runner_command.json").read_text())[0]


def test_main_returns_nonzero_when_parity_is_unqualified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_args = args(
        output_dir=tmp_path / "comparison",
        baseline_cover="legacy",
        candidate_cover="automatic",
        baseline_operator_lanes=None,
        candidate_operator_lanes=None,
        max_median_parity=0.03,
        max_median_cv=0.10,
        repetitions=1,
    )
    monkeypatch.setattr(comparison, "parse_args", lambda: run_args)
    monkeypatch.setattr(
        comparison.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            args=command, returncode=0, stdout="", stderr=""
        ),
    )
    monkeypatch.setattr(
        comparison, "classify", lambda *_args: {"controlled": False, "issues": ["drift"]}
    )
    monkeypatch.setattr(
        comparison,
        "fanout_concurrency_proof",
        lambda *_args: {"applicable": False, "passed": None},
    )
    monkeypatch.setattr(comparison, "metric_value", lambda *_args: 100.0)

    for arm in ("baseline", "candidate"):
        run_dir = run_args.output_dir / f"pair_000_{arm}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary()))
        (run_dir / "manifest.json").write_text(json.dumps(manifest()))

    assert comparison.main() == 2
