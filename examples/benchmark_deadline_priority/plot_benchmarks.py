#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Plot benchmark_deadline_priority results from CSVs produced by run_benchmarks.py.

The CSV has per-scenario columns prefixed with ref_, default_, and optionally
pinned_.  All three cases are plotted on every chart.

Usage:
    python3 plot_benchmarks.py --input-dir benchmark_results/
    python3 plot_benchmarks.py --input-dir benchmark_results/ --output-dir plots/
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCENARIO_STYLE = {
    "ref": {"color": "#888888", "label": "Hi-only (ref)"},
    "default": {"color": "#1f77b4", "label": "Default"},
    "pinned": {"color": "#2ca02c", "label": "Pinned Dispatcher"},
}

MEAN_MARKERS = {"ref": "D", "default": "o", "pinned": "^"}
P99_MARKERS = {"ref": "d", "default": "s", "pinned": "v"}


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(row, key, fallback=None):
    v = row.get(key, "-")
    if v in ("-", "", None):
        return fallback
    return float(v)


def _scenarios(rows):
    """Return list of scenarios present in the data."""
    out = ["ref", "default"]
    if any(row.get("pinned_invocation_gap_mean_us", "-") != "-" for row in rows):
        out.append("pinned")
    return out


def _sorted_points(rows, x_key, y_key):
    pts = [
        (float(r[x_key]), _safe_float(r, y_key)) for r in rows if _safe_float(r, y_key) is not None
    ]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def _set_int_xticks(ax, xs):
    int_xs = sorted({int(x) for x in xs})
    if int_xs:
        ax.set_xticks(int_xs)


# ── Invocation gap ─────────────────────────────────────────────────────────


def _invocation_plot(ax, data, x_key, xlabel, scenarios, stat):
    col_suffix = f"invocation_gap_{stat}_us"
    all_xs = []
    for s in scenarios:
        sty = SCENARIO_STYLE[s]
        xs, ys = _sorted_points(data, x_key, f"{s}_{col_suffix}")
        if not xs:
            continue
        all_xs.extend(xs)
        ax.plot(
            xs,
            ys,
            marker=MEAN_MARKERS[s],
            color=sty["color"],
            label=sty["label"],
            linewidth=2,
            markersize=6,
        )

    periods = [_safe_float(r, "hi_period_us") for r in data]
    periods = [v for v in periods if v is not None]
    if periods:
        ax.axhline(
            periods[0],
            color="#444444",
            linestyle=":",
            linewidth=2,
            label=f"Target period ({periods[0]:.0f} us)",
        )

    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    _set_int_xticks(ax, all_xs)


def plot_invocation_scaling(rows, core_counts, output_dir, scenarios):
    scaling = [r for r in rows if int(r["bg_work_us"]) == 1000]
    n = len(core_counts)
    for stat, ylabel in [
        ("mean", "Hi Invocation Gap Mean (us)"),
        ("p99", "Hi Invocation Gap P99 (us)"),
    ]:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, nc in zip(axes, core_counts, strict=True):
            subset = [r for r in scaling if int(r["cores"]) == nc]
            _invocation_plot(ax, subset, "bg_per_core", "Bg ops per core", scenarios, stat)
            ax.set_title(f"{nc} core{'s' if nc > 1 else ''}")
            if ax is axes[0]:
                ax.set_ylabel(ylabel)
        fig.suptitle(
            f"Hi Invocation Gap {stat} vs Bg Scaling\n(SCHED_DEADLINE, bg_work = 1000 us)",
            fontsize=13,
            y=1.02,
        )
        fig.tight_layout()
        fname = os.path.join(output_dir, f"dl_invocation_scaling_{stat}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}")


def plot_invocation_work(rows, output_dir, scenarios):
    work = [r for r in rows if int(r["cores"]) == 1 and int(r["bg_per_core"]) == 1]
    for stat, ylabel in [
        ("mean", "Hi Invocation Gap Mean (us)"),
        ("p99", "Hi Invocation Gap P99 (us)"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        _invocation_plot(ax, work, "bg_work_us", "Bg Work Duration (us)", scenarios, stat)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Hi Invocation Gap {stat} vs Bg Work\n(SCHED_DEADLINE, 1 core, 1 bg/core)")
        fig.tight_layout()
        fname = os.path.join(output_dir, f"dl_invocation_work_{stat}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}")


# ── Overrun ────────────────────────────────────────────────────────────────


def _overrun_plot(ax, data, x_key, xlabel, scenarios):
    all_xs = []
    for s in scenarios:
        sty = SCENARIO_STYLE[s]
        xs, ys = _sorted_points(data, x_key, f"{s}_overrun_pct")
        if not xs:
            continue
        all_xs.extend(xs)
        ax.plot(
            xs,
            ys,
            marker=MEAN_MARKERS[s],
            color=sty["color"],
            label=sty["label"],
            linewidth=2,
            markersize=6,
        )
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Overrun rate (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    _set_int_xticks(ax, all_xs)


def _missed_plot(ax, data, x_key, xlabel, scenarios):
    all_xs = []
    for s in scenarios:
        sty = SCENARIO_STYLE[s]
        xs, ys = _sorted_points(data, x_key, f"{s}_missed_pct")
        if not xs:
            continue
        all_xs.extend(xs)
        ax.plot(
            xs,
            ys,
            marker=MEAN_MARKERS[s],
            color=sty["color"],
            label=sty["label"],
            linewidth=2,
            markersize=6,
        )
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Missed rate (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    _set_int_xticks(ax, all_xs)


def plot_overrun_scaling(rows, core_counts, output_dir, scenarios):
    scaling = [r for r in rows if int(r["bg_work_us"]) == 1000]
    n = len(core_counts)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)
    for col, nc in enumerate(core_counts):
        subset = [r for r in scaling if int(r["cores"]) == nc]
        _overrun_plot(axes[0][col], subset, "bg_per_core", "Bg ops per core", scenarios)
        _missed_plot(axes[1][col], subset, "bg_per_core", "Bg ops per core", scenarios)
        axes[0][col].set_title(f"{nc} core{'s' if nc > 1 else ''}")
    axes[0][0].set_ylabel("Overrun (>period+100us) %")
    axes[1][0].set_ylabel("Missed (>1.5x period) %")
    fig.suptitle("Overruns vs Bg Scaling\n(SCHED_DEADLINE, bg_work = 1000 us)", fontsize=13, y=1.02)
    fig.tight_layout()
    fname = os.path.join(output_dir, "dl_overrun_scaling.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def plot_overrun_work(rows, output_dir, scenarios):
    work = [r for r in rows if int(r["cores"]) == 1 and int(r["bg_per_core"]) == 1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    _overrun_plot(ax1, work, "bg_work_us", "Bg Work Duration (us)", scenarios)
    ax1.set_title("Overrun (>period+100us)")
    _missed_plot(ax2, work, "bg_work_us", "Bg Work Duration (us)", scenarios)
    ax2.set_title("Missed (>1.5x period)")
    fig.suptitle("Overruns vs Bg Work\n(SCHED_DEADLINE, 1 core, 1 bg/core)", fontsize=13, y=1.02)
    fig.tight_layout()
    fname = os.path.join(output_dir, "dl_overrun_work.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ── Drift ──────────────────────────────────────────────────────────────────


def _drift_plot(ax, data, x_key, xlabel, scenarios):
    all_xs = []
    for s in scenarios:
        sty = SCENARIO_STYLE[s]
        xs, y_mean = _sorted_points(data, x_key, f"{s}_drift_mean_us")
        _, y_p99 = _sorted_points(data, x_key, f"{s}_drift_p99_us")
        if not xs:
            continue
        all_xs.extend(xs)
        ax.plot(
            xs,
            y_mean,
            marker=MEAN_MARKERS[s],
            color=sty["color"],
            label=f"{sty['label']} mean",
            linewidth=2,
            markersize=6,
        )
        ax.plot(
            xs,
            y_p99,
            marker=P99_MARKERS[s],
            color=sty["color"],
            linestyle="--",
            label=f"{sty['label']} p99",
            linewidth=1.5,
            markersize=5,
        )
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Drift (us)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)
    _set_int_xticks(ax, all_xs)


def plot_drift_scaling(rows, core_counts, output_dir, scenarios):
    scaling = [r for r in rows if int(r["bg_work_us"]) == 1000]
    n = len(core_counts)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, nc in zip(axes, core_counts, strict=True):
        subset = [r for r in scaling if int(r["cores"]) == nc]
        _drift_plot(ax, subset, "bg_per_core", "Bg ops per core", scenarios)
        ax.set_title(f"{nc} core{'s' if nc > 1 else ''}")
    fig.suptitle("Drift vs Reference\n(SCHED_DEADLINE, bg_work = 1000 us)", fontsize=13, y=1.02)
    fig.tight_layout()
    fname = os.path.join(output_dir, "dl_drift_scaling.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def plot_drift_work(rows, output_dir, scenarios):
    work = [r for r in rows if int(r["cores"]) == 1 and int(r["bg_per_core"]) == 1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _drift_plot(ax, work, "bg_work_us", "Bg Work Duration (us)", scenarios)
    ax.set_title("Drift vs Reference\n(SCHED_DEADLINE, 1 core, 1 bg/core)")
    fig.tight_layout()
    fname = os.path.join(output_dir, "dl_drift_work.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# ── main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark_deadline_priority sweep results.",
    )
    parser.add_argument(
        "--input-dir", default="benchmark_results", help="Directory containing deadline_results.csv"
    )
    parser.add_argument(
        "--output-dir", default="benchmark_results", help="Directory for PNG output"
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    dl_csv = os.path.join(args.input_dir, "deadline_results.csv")
    dl_rows = read_csv(dl_csv)

    if dl_rows:
        scenarios = _scenarios(dl_rows)
        print(f"\nDEADLINE plots ({len(dl_rows)} rows, scenarios: {scenarios}):")
        core_counts = sorted({int(r["cores"]) for r in dl_rows if int(r["bg_work_us"]) == 1000})
        if core_counts:
            plot_invocation_scaling(dl_rows, core_counts, output_dir, scenarios)
            plot_overrun_scaling(dl_rows, core_counts, output_dir, scenarios)
            plot_drift_scaling(dl_rows, core_counts, output_dir, scenarios)
        work_rows = [r for r in dl_rows if int(r["cores"]) == 1 and int(r["bg_per_core"]) == 1]
        if len({int(r["bg_work_us"]) for r in work_rows}) > 1:
            plot_invocation_work(dl_rows, output_dir, scenarios)
            plot_overrun_work(dl_rows, output_dir, scenarios)
            plot_drift_work(dl_rows, output_dir, scenarios)
    else:
        print(f"\nSkipping plots ({dl_csv} not found or empty)")

    print("\nDone.")


if __name__ == "__main__":
    main()
