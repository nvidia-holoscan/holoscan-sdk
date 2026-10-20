#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run robot-control-loop benchmark arms in interleaved rounds and summarize them.

An arm names a binary (possibly from another build), its native arguments, and extra environment
(for example LD_PRELOAD). Every round runs each arm once, in a fresh process, in a seeded
random order, so slow drift of the host affects every arm alike. Results, logs, and a manifest
are kept per run; the summary reports per-arm medians and ranges of per-run percentiles, and
deltas against a reference arm.
"""

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
from pathlib import Path

SCHEMA = "holoscan.control_loop.v1"
KPIS = (
    "control_loop_latency",
    "active_path",
    "entry_latency",
    "control_period",
    "control_period_error",
    "target_pose_response_latency",
    "target_delivery",
    "hop_read_to_motion",
    "hop_motion_to_safety",
    "hop_safety_to_write",
)
PERCENTILES = ("p50", "p90", "p99", "p999", "p9999", "max")
COUNTS = ("late", "coalesced", "missed_freshness", "dropped_publications")


def validate_result(result):
    """Reject results that are invalid, incomplete, or claim unsupported percentiles."""
    if result.get("schema") != SCHEMA:
        raise ValueError("unexpected result schema")
    if result.get("valid") is not True:
        raise ValueError("benchmark reported an invalid run")
    if result.get("diagnostics") != []:
        raise ValueError("Runtime reported diagnostics")
    samples = result["samples"]
    counts = result["counts"]
    if counts["measured_ticks"] != samples or counts["missing_ticks"] != 0:
        raise ValueError("measured tick count mismatch")
    for name in ("control_loop_latency", "active_path", "entry_latency"):
        timing = result["kpis_ns"][name]
        if timing["samples"] != samples or not 0 <= timing["p50"] <= timing["max"]:
            raise ValueError(f"missing or invalid {name} samples")
        if (samples >= 100_000) != (timing["p9999"] is not None):
            raise ValueError("unsupported P99.99 sample count")


def load_arms(path):
    """Read {"reference": name, "arms": [{"name", "binary", "args", "env"}]} and check it."""
    document = json.loads(Path(path).read_text())
    arms = document.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("arms file needs a non-empty 'arms' list")
    names = [arm.get("name") for arm in arms]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(
        names
    ):
        raise ValueError("every arm needs a unique name")
    for arm in arms:
        args = arm.setdefault("args", [])
        env = arm.setdefault("env", {})
        if (
            not isinstance(arm.get("binary"), str)
            or not isinstance(args, list)
            or not all(isinstance(arg, str) for arg in args)
            or not isinstance(env, dict)
            or not all(isinstance(item, str) for pair in env.items() for item in pair)
        ):
            raise ValueError(
                f"arm {arm['name']} needs a binary, a list of string args, and a string env map"
            )
        if any(arg == "--output" or arg.startswith("--output=") for arg in args):
            raise ValueError(f"arm {arm['name']} must not set --output")
    reference = document.get("reference", names[0])
    if reference not in names:
        raise ValueError("reference must name an arm")
    return reference, arms


def round_orders(arm_names, rounds, seed):
    """Return one shuffled order of every arm per round, reproducible from the seed."""
    generator = random.Random(seed)
    orders = []
    for _ in range(rounds):
        order = list(arm_names)
        generator.shuffle(order)
        orders.append(order)
    return orders


def summarize_arm(results):
    """Median and range of per-run percentiles for each KPI, plus summed correctness counts."""
    summary = {"runs": len(results), "kpis_ns": {}, "counts": {}}
    for kpi in KPIS:
        entry = {}
        for percentile in PERCENTILES:
            values = [
                result["kpis_ns"][kpi][percentile]
                for result in results
                if result["kpis_ns"].get(kpi, {}).get(percentile) is not None
            ]
            if values:
                entry[percentile] = {
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                }
        if entry:
            summary["kpis_ns"][kpi] = entry
    for name in COUNTS:
        summary["counts"][name] = sum(result["counts"].get(name, 0) for result in results)
    counted = [result for result in results if result["phase_counters"]["enabled"]]
    if counted:
        # Counters cover warmup too. Per-partition entries are summed by lane role.
        ticks = sum(result["samples"] + result["warmup"] for result in counted)
        totals, partitions = {}, {}
        for result in counted:
            for name, value in result["phase_counters"].items():
                if is_count(value):
                    totals[name] = totals.get(name, 0) + value
            for partition in result["phase_counters"].get("partitions", []):
                entry = partitions.setdefault(partition.get("role", "other"), {})
                for name, value in partition.items():
                    if is_count(value) and name not in ("index", "owner_tid"):
                        entry[name] = entry.get(name, 0) + value
        summary["phase_counters_per_tick"] = {name: value / ticks for name, value in totals.items()}
        summary["partition_counters_per_tick"] = {
            role: {name: value / ticks for name, value in entry.items()}
            for role, entry in partitions.items()
        }
    return summary


def is_count(value):
    """Integer counters only; lists, flags, and identifiers are summarized elsewhere."""
    return isinstance(value, int) and not isinstance(value, bool)


def deltas(summaries, reference):
    """Differences of median per-run P50 and P99 against the reference arm, in ns and ratio."""
    base = summaries[reference]["kpis_ns"]
    result = {}
    for name, summary in summaries.items():
        if name == reference:
            continue
        entry = {}
        for kpi in ("control_loop_latency", "active_path", "entry_latency"):
            for percentile in ("p50", "p99"):
                ours = summary["kpis_ns"].get(kpi, {}).get(percentile)
                theirs = base.get(kpi, {}).get(percentile)
                if ours and theirs and theirs["median"]:
                    entry[f"{kpi}.{percentile}"] = {
                        "delta_ns": ours["median"] - theirs["median"],
                        "ratio": ours["median"] / theirs["median"],
                    }
        result[name] = entry
    return result


def markdown(summaries, reference):
    """A compact table of median per-run percentiles (µs) for the headline KPIs."""
    lines = [
        "| Arm | Loop P50 | Loop P99 | Loop P99.9 | Loop max | Active P50 | Entry P50 | Late |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def cell(summary, kpi, percentile):
        value = summary["kpis_ns"].get(kpi, {}).get(percentile)
        return f"{value['median'] / 1000:.2f}" if value else "–"

    for name, summary in summaries.items():
        marker = " (reference)" if name == reference else ""
        lines.append(
            f"| {name}{marker} | {cell(summary, 'control_loop_latency', 'p50')} | "
            f"{cell(summary, 'control_loop_latency', 'p99')} | "
            f"{cell(summary, 'control_loop_latency', 'p999')} | "
            f"{cell(summary, 'control_loop_latency', 'max')} | "
            f"{cell(summary, 'active_path', 'p50')} | {cell(summary, 'entry_latency', 'p50')} | "
            f"{summary['counts']['late']} |"
        )
    return "\n".join(lines) + "\n"


def git_state(source_root):
    """Record the local source identity without requiring a remote or changing the checkout."""
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source_root, text=True
    )
    return {"revision": revision, "tracked_changes": changes}


def source_state(path):
    """Allow host-captured provenance when the container mount excludes Git metadata."""
    if path is None:
        try:
            return git_state(Path(__file__).resolve().parents[2])
        except (subprocess.SubprocessError, OSError):
            return {"revision": None, "tracked_changes": None}
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=Path, required=True, help="arms JSON file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--order-seed", type=int, default=1)
    parser.add_argument("--process-timeout-sec", type=float, default=600)
    parser.add_argument("--source-state", type=Path, help="host-captured Git state JSON")
    args = parser.parse_args()
    if args.rounds < 1 or args.process_timeout_sec <= 0:
        parser.error("use positive rounds and timeout")
    reference, arms = load_arms(args.arms)
    by_name = {arm["name"]: arm for arm in arms}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    orders = round_orders(list(by_name), args.rounds, args.order_seed)
    manifest = {
        "source": source_state(args.source_state),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "reference": reference,
        "arms": arms,
        "orders": orders,
        "runs": [],
    }
    results = {name: [] for name in by_name}
    failures = []
    for round_index, order in enumerate(orders):
        for name in order:
            arm = by_name[name]
            stem = f"{name}-round{round_index}"
            result_path = (args.output_dir / f"{stem}.json").resolve()
            command = [
                str(Path(arm["binary"]).resolve()),
                *arm["args"],
                "--output",
                str(result_path),
            ]
            manifest["runs"].append({"arm": name, "round": round_index, "command": command})
            (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            environment = dict(os.environ, **arm["env"])
            try:
                with (args.output_dir / f"{stem}.log").open("w") as log:
                    subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=args.process_timeout_sec,
                        env=environment,
                    )
                result = json.loads(result_path.read_text())
                validate_result(result)
                results[name].append(result)
            except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
                failures.append({"arm": name, "round": round_index, "error": str(error)})
    summaries = {name: summarize_arm(runs) for name, runs in results.items() if runs}
    summary = {
        "reference": reference,
        "arms": summaries,
        "deltas": deltas(summaries, reference) if reference in summaries else {},
        "failures": failures,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = markdown(summaries, reference)
    (args.output_dir / "summary.md").write_text(table)
    print(table)
    print(json.dumps({"failures": failures}))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
