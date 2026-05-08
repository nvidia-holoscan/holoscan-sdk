#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Run benchmark_deadline_priority DEADLINE sweeps, parse stdout, write CSVs.

Sweeps:
  DEADLINE:
    Scaling  - cores 1-3, bg_per_core 1-5, bg_work_us=1000 (default)
    Work     - cores 1,   bg_per_core=1,   bg_work_us 1000-3000 step 500

Core IDs start from --core-start (default 4).

Usage (requires root for DEADLINE):
    sudo python3 run_benchmarks.py --binary ./benchmark_deadline_priority
    sudo python3 run_benchmarks.py --binary ./benchmark_deadline_priority --core-start 0
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

METRIC_KEYS = [
    "wall_s",
    "hi_ticks",
    "bg_ticks",
    "throughput_hz",
    "dispatch_gap_mean_us",
    "dispatch_gap_p99_us",
    "invocation_gap_mean_us",
    "invocation_gap_p99_us",
    "overhead_mean_us",
    "overhead_p99_us",
    "overhead_max_us",
    "overrun_pct",
    "missed_pct",
    "drift_late_count",
    "drift_total_samples",
    "drift_mean_us",
    "drift_p99_us",
    "drift_max_us",
]

ROW_TO_KEY = {
    "Wall time (s)": "wall_s",
    "Hi ticks": "hi_ticks",
    "Bg ticks": "bg_ticks",
    "Throughput (ops/s)": "throughput_hz",
    "Dispatch gap mean (us)": "dispatch_gap_mean_us",
    "Dispatch gap p99 (us)": "dispatch_gap_p99_us",
    "Invocation gap mean (us)": "invocation_gap_mean_us",
    "Invocation gap p99 (us)": "invocation_gap_p99_us",
    "Overhead mean (us)": "overhead_mean_us",
    "Overhead p99 (us)": "overhead_p99_us",
    "Overhead max (us)": "overhead_max_us",
    "Overrun (>period+100us) rate": "overrun_pct",
    "Missed (>1.5x period) rate": "missed_pct",
    "Drift late count": "drift_late_count",
    "Drift total samples": "drift_total_samples",
    "Drift mean (us)": "drift_mean_us",
    "Drift p99 (us)": "drift_p99_us",
    "Drift max (us)": "drift_max_us",
}


def run_cmd(
    binary, cores, core_start, lo_per_core=1, lo_work_us=1000, dispatcher_pin_core=None, timeout=600
):
    core_ids = [str(core_start + i) for i in range(cores)]
    core_list = ",".join(core_ids)
    cmd = ["taskset", "-c", core_list, binary]
    cmd += ["-n", str(lo_per_core), "-w", str(lo_work_us)]
    if dispatcher_pin_core is not None:
        cmd += ["-p", str(dispatcher_pin_core)]
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    combined = result.stdout + "\n" + result.stderr
    if "SCHED_DEADLINE requires root" in combined or "CAP_SYS_ADMIN" in combined:
        print(
            "\nERROR: SCHED_DEADLINE requires root / CAP_SYS_ADMIN. Re-run with sudo.",
            file=sys.stderr,
        )
        sys.exit(1)
    if result.returncode != 0:
        print(f"  WARNING: exit code {result.returncode}", file=sys.stderr)
        if result.stderr:
            for line in result.stderr.strip().splitlines()[:5]:
                print(f"    stderr: {line}", file=sys.stderr)
    return combined


def parse_table(output):
    """Parse the comparison table from benchmark output.

    Returns a dict mapping column name -> {metric_key: value_str}.
    """
    results_idx = output.find("RESULTS")
    if results_idx < 0:
        return None, 0.0
    text = output[results_idx:]

    m = re.search(r"hi target period:\s*([\d.]+)\s*us", text)
    hi_period_us = float(m.group(1)) if m else 0.0

    header_match = re.search(r"^(Metric\s*\|.+)$", text, re.MULTILINE)
    if not header_match:
        return None, hi_period_us

    header_line = header_match.group(1)
    col_names = [c.strip() for c in header_line.split("|")[1:] if c.strip()]

    columns = {name: {} for name in col_names}

    for line in text[header_match.end() :].splitlines():
        line = line.rstrip()
        if not line or line.startswith("-") or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        row_label = parts[0]
        key = ROW_TO_KEY.get(row_label)
        if not key:
            continue
        values = parts[1:]
        for i, name in enumerate(col_names):
            if i < len(values):
                raw = values[i].rstrip("%").strip()
                columns[name][key] = raw
            else:
                columns[name][key] = "-"

    return columns, hi_period_us


def make_csv_fields(scenarios):
    """Build CSV field list: config fields + per-scenario metric fields."""
    fields = ["cores", "bg_per_core", "bg_work_us", "hi_period_us"]
    for s in scenarios:
        fields.extend(f"{s}_{mk}" for mk in METRIC_KEYS)
    return fields


def main():
    parser = argparse.ArgumentParser(
        description="Sweep benchmark_deadline_priority across DEADLINE configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--binary",
        default="./build-debug-x86_64/examples/benchmark_deadline_priority/cpp/benchmark_deadline_priority",
        help="Path to benchmark_deadline_priority binary",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory for CSV output (default: benchmark_results/)",
    )
    parser.add_argument(
        "--core-start", type=int, default=4, help="First core ID to use (default: 4)"
    )
    parser.add_argument(
        "--dispatcher-pin-core",
        type=int,
        default=1,
        help="Pin dispatcher to this core with SCHED_FIFO 99 (-p flag)",
    )
    parser.add_argument(
        "--dl-cores",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Core counts for DEADLINE scaling sweep",
    )
    parser.add_argument(
        "--dl-bg-range",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="bg_per_core values for DEADLINE scaling sweep",
    )
    parser.add_argument(
        "--dl-work-range",
        nargs="+",
        type=int,
        default=[500, 1000, 1500, 2000, 2500],
        help="bg_work_us values for DEADLINE work sweep (1 core)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        print(f"ERROR: binary not found: {args.binary}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    dl_configs = set()
    for c in args.dl_cores:
        for n in args.dl_bg_range:
            dl_configs.add((c, n, 1000))
    for w in args.dl_work_range:
        dl_configs.add((1, 1, w))
    dl_configs = sorted(dl_configs)

    total = len(dl_configs)
    done = 0

    def progress(label):
        nonlocal done
        done += 1
        pct = done * 100 // total if total else 100
        filled = 30 * done // total if total else 30
        bar = "#" * filled + "-" * (30 - filled)
        print(f"  [{bar}] {done}/{total} ({pct}%) {label}\n", flush=True)

    scenarios = ["ref", "default"]
    if args.dispatcher_pin_core is not None:
        scenarios.append("pinned")

    col_to_scenario = {
        "Hi-only (ref)": "ref",
        "Default": "default",
        "Pinned Dispatcher": "pinned",
    }

    print(f"\nTotal runs: {total}")
    print(f"Core IDs start from {args.core_start}\n")

    if dl_configs:
        dl_csv = os.path.join(args.output_dir, "deadline_results.csv")
        fields = make_csv_fields(scenarios)
        print(f"{'=' * 60}")
        print("SCHED_DEADLINE SWEEP")
        print(f"{'=' * 60}")
        with open(dl_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            f.flush()
            for cores, bg_n, bg_w in dl_configs:
                print(f"\n[DEADLINE] {cores} core(s), {bg_n} bg/core, bg_work={bg_w} us")
                t0 = time.monotonic()
                try:
                    output = run_cmd(
                        args.binary,
                        cores,
                        args.core_start,
                        lo_per_core=bg_n,
                        lo_work_us=bg_w,
                        dispatcher_pin_core=args.dispatcher_pin_core,
                    )
                except subprocess.TimeoutExpired:
                    print("  ERROR: timeout", file=sys.stderr)
                    progress(f"DL {cores}c/{bg_n}n/{bg_w}w TIMEOUT")
                    continue
                elapsed = time.monotonic() - t0
                columns, hi_period = parse_table(output)
                if columns is None:
                    print(f"  WARNING: no results parsed ({elapsed:.1f}s)")
                    progress(f"DL {cores}c/{bg_n}n/{bg_w}w ({elapsed:.1f}s)")
                    continue

                row = {
                    "cores": cores,
                    "bg_per_core": bg_n,
                    "bg_work_us": bg_w,
                    "hi_period_us": hi_period,
                }
                for col_name, metrics in columns.items():
                    scenario = col_to_scenario.get(col_name)
                    if not scenario or scenario not in scenarios:
                        continue
                    for mk in METRIC_KEYS:
                        row[f"{scenario}_{mk}"] = metrics.get(mk, "-")

                parts = []
                for s in scenarios:
                    v = row.get(f"{s}_invocation_gap_mean_us", "-")
                    parts.append(f"{s}={v}")
                print(f"    invoc_mean: {', '.join(parts)}")
                writer.writerow(row)
                f.flush()
                progress(f"DL {cores}c/{bg_n}n/{bg_w}w ({elapsed:.1f}s)")
        print(f"\n  -> {dl_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
