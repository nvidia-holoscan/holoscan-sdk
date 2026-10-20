#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run and retain independent concurrent-fanout repetitions in the SDK container."""

import argparse
import json
import platform
import subprocess
from pathlib import Path


def validate_result(result):
    """Reject incomplete geometry, missing timing evidence, and failed natural completion."""
    if result.get("schema") != "holoscan.fanout_scaling.v1":
        raise ValueError("unexpected result schema")
    if result.get("valid") is not True or result.get("completed") is not True:
        raise ValueError("benchmark did not validate and complete naturally")
    if result.get("diagnostics") != []:
        raise ValueError("Runtime reported diagnostics")
    producers, fanout = result["producer_count"], result["fanout"]
    samples, warmup = result["samples"], result["warmup"]
    if len(result["producers"]) != producers or len(result["sinks"]) != producers * fanout:
        raise ValueError("missing producer or sink records")
    for index, producer in enumerate(result["producers"]):
        if producer["index"] != index or producer["publications"] != samples + warmup:
            raise ValueError("producer publication count mismatch")
        for name in ("latency_ns", "emit_ns"):
            timing = producer[name]
            if timing["samples"] != samples or not 0 <= timing["p50"] <= timing["max"]:
                raise ValueError("missing or invalid timing samples")
            if (samples >= 100_000) != (timing["p9999"] is not None):
                raise ValueError("unsupported P99.99 sample count")
    for index, sink in enumerate(result["sinks"]):
        if sink["index"] != index or sink["deliveries"] != samples + warmup:
            raise ValueError("sink delivery count mismatch")
    if result["measured_window_ns"] <= 0 or result["publications_per_second"] <= 0:
        raise ValueError("invalid measured window")
    lanes = producers + (fanout if result["shared_destinations"] else producers * fanout)
    if len(result["lane_startup"]) != lanes:
        raise ValueError("missing lane readback")
    if result["phase_counters"]["enabled"]:
        counters = result["phase_counters"]
        scoped = counters.get("cut_scoped_retries", 0)
        if counters["cut_drains"] + counters["cut_skips"] + scoped != counters["cuts"]:
            raise ValueError("inconsistent activation-cut counters")


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
        return git_state(Path(__file__).resolve().parents[2])
    result = json.loads(path.read_text())
    if not isinstance(result.get("revision"), str) or not result["revision"]:
        raise ValueError("source state requires a revision")
    if not isinstance(result.get("tracked_changes"), str):
        raise ValueError("source state requires tracked_changes")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--process-timeout-sec", type=float, default=60)
    parser.add_argument("--source-state", type=Path, help="host-captured Git state JSON")
    args, native = parser.parse_known_args()
    if args.repetitions < 1 or args.process_timeout_sec <= 0 or "--output" in native:
        parser.error("use positive bounds and let the runner own --output")
    try:
        source = source_state(args.source_state)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        parser.error(f"cannot record source state; pass --source-state JSON: {error}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source": source,
        "kernel": platform.release(),
        "machine": platform.machine(),
        "commands": [],
        "process_timeout_sec": args.process_timeout_sec,
    }
    failures, results = [], []
    for index in range(args.repetitions):
        result_path = (args.output_dir / f"result-{index}.json").resolve()
        command = [str(args.binary.resolve()), *native, "--output", str(result_path)]
        manifest["commands"].append(command)
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        try:
            with (args.output_dir / f"run-{index}.log").open("w") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=args.process_timeout_sec,
                )
            result = json.loads(result_path.read_text())
            validate_result(result)
            results.append(result)
        except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError) as error:
            failures.append({"repetition": index, "error": str(error)})
    summary = {"passed": len(results), "failed": failures}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
