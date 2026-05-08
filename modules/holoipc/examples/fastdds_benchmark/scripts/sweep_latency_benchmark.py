#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run run_latency_benchmark.sh over many payload sizes (host + accel + optional accel eager),
# plot PNG, write CSV, and generate REPORT.md / REPORT.html — all under a date/time subfolder
# by default.
#
# Requires: Python 3.8+, matplotlib (`pip install matplotlib`)
#
# Usage (from directory containing fastdds_benchmark and copied scripts):
#   python3 scripts/sweep_latency_benchmark.py
#   python3 sweep_latency_benchmark.py -b ./fastdds_benchmark
#   python3 sweep_latency_benchmark.py -b ./fastdds_benchmark -o ./my_run_dir

from __future__ import annotations

import argparse
import csv
import html
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory
except ImportError:
    plt = None
    blended_transform_factory = None

# Default sweep: small → large, including ~RGB frame sizes (8-bit, 3 B/px).
DEFAULT_SIZES_BYTES = (
    1024,
    4 * 1024,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    6 * 1024 * 1024,
    11 * 1024 * 1024,
    15 * 1024 * 1024,
    25 * 1024 * 1024,
)

# Vertical guides on the plot (size_bytes, label) — uncompressed RGB ballpark.
VIDEO_MARKERS = (
    (6 * 1024 * 1024, "FHD (~6 MB)"),
    (11 * 1024 * 1024, "QHD (~11 MB)"),
    (15 * 1024 * 1024, "5MP (~15 MB)"),
    (25 * 1024 * 1024, "UHD (~25 MB)"),
)

MEAN_RE = re.compile(r"mean:\s*([\d,]+)\s+ns\s+\(\s*([\d.]+)\s*µs\)")
# First "--- title (n=N) ---" block in run_latency_benchmark.sh output (primary stats).
STATS_SECTION_HDR = re.compile(r"^\s+--- .+ \(n=\d+\) ---\s*$")

REPORT_MD = "REPORT.md"
REPORT_HTML = "REPORT.html"
PLOT_PNG = "latency_sweep.png"
RESULTS_CSV = "results.csv"


def payload_label(z: int) -> str:
    """Human-readable payload size (prefer MiB from 512 KiB upward)."""
    if z >= 512 * 1024:
        return f"{z / (1024 * 1024):.1f} MiB"
    return f"{z / 1024:.1f} KiB"


def default_report_dir() -> Path:
    """Subfolder name encodes local date and time when the sweep starts."""
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
    return Path(f"latency_sweep_{stamp}")


def _read_linux_cpu_model() -> str | None:
    hardware: str | None = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                line_l = line.lower()
                if "model name" in line_l and ":" in line:
                    return line.split(":", 1)[1].strip()
                if "cpu model" in line_l and ":" in line and "revision" not in line_l:
                    return line.split(":", 1)[1].strip()
                if line_l.startswith("hardware") and ":" in line:
                    hardware = line.split(":", 1)[1].strip()
    except OSError:
        pass
    return hardware


def _read_dmi_product_name() -> str | None:
    """Product name from SMBIOS/DMI (common on x86; often missing on ARM)."""
    try:
        raw = Path("/sys/class/dmi/id/product_name").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return raw if raw else None


def _read_linux_mem_total_gib() -> str | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    kb = int(parts[1])
                    gib = kb / (1024 * 1024)
                    return f"{gib:.2f} GiB (MemTotal {kb} kB)"
    except (OSError, ValueError, IndexError):
        pass
    return None


def _nvidia_smi_parse_output(stdout: str) -> tuple[str | None, str | None]:
    """Parse CSV lines from nvidia-smi --query-gpu=...; return (gpu summary block, cuda version)."""
    lines_out: list[str] = []
    cuda_ver: str | None = None
    for raw in stdout.strip().splitlines():
        try:
            fields = next(csv.reader([raw]))
        except StopIteration:
            continue
        if len(fields) >= 5:
            idx, name, drv, vram, cv = (
                fields[0],
                fields[1],
                fields[2],
                fields[3],
                fields[4],
            )
            if cuda_ver is None and cv.strip():
                cuda_ver = cv.strip()
            lines_out.append(f"[{idx}] {name} | driver {drv} | VRAM {vram}")
        elif len(fields) >= 4:
            idx, name, drv, vram = fields[0], fields[1], fields[2], fields[3]
            lines_out.append(f"[{idx}] {name} | driver {drv} | VRAM {vram}")
        elif raw.strip():
            lines_out.append(raw.strip())
    block = "\n".join(lines_out) if lines_out else None
    return block, cuda_ver


def _nvidia_smi_gpu_and_cuda() -> tuple[str | None, str | None]:
    """GPU lines and driver-reported CUDA version (one nvidia-smi when possible)."""
    for query in (
        "index,name,driver_version,memory.total,cuda_version",
        "index,name,driver_version,memory.total",
    ):
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=" + query,
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None, None
        if proc.returncode == 0 and proc.stdout.strip():
            gpu_block, cuda_v = _nvidia_smi_parse_output(proc.stdout)
            if cuda_v is None and "cuda_version" not in query:
                cuda_v = _cuda_version_from_nvsmi_banner()
            return gpu_block, cuda_v
    return None, None


def _cuda_version_from_nvsmi_banner() -> str | None:
    """Parse 'CUDA Version: X.Y' from full nvidia-smi text output."""
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if "CUDA Version:" in line:
            tail = line.split("CUDA Version:", 1)[1].strip().split()
            return tail[0] if tail else None
    return None


def _cuda_toolkit_version_nvcc() -> str | None:
    """Installed CUDA toolkit version from nvcc, if on PATH."""
    try:
        proc = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        line_l = line.lower()
        if "release" in line_l:
            m = re.search(r"release\s+([\d.]+)", line, flags=re.IGNORECASE)
            if m:
                return m.group(1)
    return None


def _cuda_version_summary(cuda_from_driver: str | None) -> str | None:
    parts: list[str] = []
    if cuda_from_driver:
        parts.append(f"driver {cuda_from_driver} (nvidia-smi)")
    toolkit = _cuda_toolkit_version_nvcc()
    if toolkit:
        parts.append(f"toolkit {toolkit} (nvcc)")
    if not parts:
        return None
    return "; ".join(parts)


def collect_system_summary() -> list[tuple[str, str]]:
    """Best-effort host summary for REPORT.md / REPORT.html (no extra Python deps)."""
    rows: list[tuple[str, str]] = []
    product = _read_dmi_product_name()
    if product:
        rows.append(("Product name (DMI)", product))
    rows.append(
        (
            "Platform",
            f"{platform.system()} {platform.release()} ({platform.machine()})",
        )
    )
    cpu_detail = _read_linux_cpu_model()
    if cpu_detail:
        rows.append(("CPU model (/proc/cpuinfo)", cpu_detail))

    ncpu = os.cpu_count()
    if ncpu is not None:
        rows.append(("Logical CPUs", str(ncpu)))

    mem = _read_linux_mem_total_gib()
    if mem:
        rows.append(("Memory", mem))

    gpu, cuda_drv = _nvidia_smi_gpu_and_cuda()
    rows.append(("GPU (nvidia-smi)", gpu if gpu else "(not available or not run)"))

    cuda_s = _cuda_version_summary(cuda_drv)
    rows.append(("CUDA", cuda_s if cuda_s else "(not available)"))

    rows.append(("Python", platform.python_version()))
    return rows


def write_report_md(
    path: Path,
    *,
    generated_at_utc: str,
    generated_at_local: str,
    runner: Path,
    samples: int,
    sizes: tuple[int, ...],
    binary: str | None,
    no_accel: bool,
    plot_file: str,
    csv_file: str,
    host_ns: list[int | None],
    accel_ns: list[int | None],
    eager_ns: list[int | None],
    include_eager: bool,
    system_rows: list[tuple[str, str]],
) -> None:
    lines = [
        "# fastdds_benchmark — latency sweep",
        "",
        f"- **Generated (local):** {generated_at_local}",
        f"- **Generated (UTC):** {generated_at_utc}",
        f"- **Runner:** `{runner}`",
        f"- **Samples per size / run:** {samples} (`-w 1` → mean over indices ≥ 1)",
        f"- **Payload sizes:** {len(sizes)} points from {sizes[0]} to {sizes[-1]} bytes",
        f"- **Binary:** `{binary or '(default from runner / env)'}`",
        "- **Accel path:** "
        + ("disabled (`--no-accel`)" if no_accel else "enabled (AccelBuffer / CUDA IPC)"),
        "- **Accel eager sweep:** "
        + (
            "off (`--no-eager` or `--no-accel`)"
            if not include_eager
            else "on (subscriber `acquire_pointer_eager`, runner `-e`)"
        ),
        "",
        "## System",
        "",
    ]
    for label, val in system_rows:
        if "\n" in val:
            lines.append(f"- **{label}:**")
            lines.extend(f"  - {sub}" for sub in val.split("\n"))
        else:
            lines.append(f"- **{label}:** {val}")
    lines.extend(
        [
            "",
            "## Plot",
            "",
            f"![Mean one-way latency vs payload size]({plot_file})",
            "",
            "## Mean latency (ms), indices ≥ 1",
            "",
        ]
    )
    if include_eager:
        lines.extend(
            [
                "| Payload | Size (bytes) | Host (ms) | Accel (ms) | Accel eager (ms) |",
                "|--------:|-------------:|----------:|-----------:|-----------------:|",
            ]
        )
        for i, z in enumerate(sizes):
            h = host_ns[i]
            a = accel_ns[i]
            e = eager_ns[i]
            h_s = f"{h / 1e6:.6f}" if h is not None else ""
            a_s = f"{a / 1e6:.6f}" if a is not None else ""
            e_s = f"{e / 1e6:.6f}" if e is not None else ""
            label = payload_label(z)
            lines.append(f"| {label} | {z} | {h_s} | {a_s} | {e_s} |")
    else:
        lines.extend(
            [
                "| Payload | Size (bytes) | Host (ms) | Accel (ms) |",
                "|--------:|-------------:|----------:|-----------:|",
            ]
        )
        for i, z in enumerate(sizes):
            h = host_ns[i]
            a = accel_ns[i]
            h_s = f"{h / 1e6:.6f}" if h is not None else ""
            a_s = "" if no_accel else (f"{a / 1e6:.6f}" if a is not None else "")
            label = payload_label(z)
            lines.append(f"| {label} | {z} | {h_s} | {a_s} |")
    lines.extend(
        [
            "",
            "## Files in this folder",
            "",
            f"- `{plot_file}` — log–log chart (host / accel / optional accel eager mean latency)",
            f"- `{csv_file}` — machine-readable rows (`size_bytes`, `flow`, `mean_ns`)",
            f"- `{REPORT_MD}` — this document (Markdown)",
            f"- `{REPORT_HTML}` — same report as HTML (open in a browser)",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_html(
    path: Path,
    *,
    generated_at_utc: str,
    generated_at_local: str,
    runner: Path,
    samples: int,
    sizes: tuple[int, ...],
    binary: str | None,
    no_accel: bool,
    plot_file: str,
    csv_file: str,
    host_ns: list[int | None],
    accel_ns: list[int | None],
    eager_ns: list[int | None],
    include_eager: bool,
    system_rows: list[tuple[str, str]],
) -> None:
    runner_s = html.escape(str(runner), quote=True)
    binary_disp = binary or "(default from runner / env)"
    binary_s = html.escape(binary_disp, quote=True)
    accel_s = html.escape(
        "disabled (--no-accel)" if no_accel else "enabled (AccelBuffer / CUDA IPC)",
        quote=True,
    )
    plot_esc = html.escape(plot_file, quote=True)

    sys_items: list[str] = []
    for label, val in system_rows:
        lab_esc = html.escape(label, quote=True)
        if "\n" in val:
            sub = "".join(
                f"<li>{html.escape(s, quote=True)}</li>" for s in val.split("\n") if s.strip()
            )
            sys_items.append(f"<li><strong>{lab_esc}:</strong><ul>{sub}</ul></li>")
        else:
            sys_items.append(f"<li><strong>{lab_esc}:</strong> {html.escape(val, quote=True)}</li>")
    system_html = "\n    ".join(sys_items)

    eager_on_s = html.escape(
        "on (subscriber --eager)" if include_eager else "off (--no-eager or --no-accel)",
        quote=True,
    )
    table_rows: list[str] = []
    for i, z in enumerate(sizes):
        h = host_ns[i]
        a = accel_ns[i]
        h_s = f"{h / 1e6:.6f}" if h is not None else ""
        a_s = "" if no_accel else (f"{a / 1e6:.6f}" if a is not None else "")
        label = payload_label(z)
        label_esc = html.escape(label, quote=True)
        if include_eager:
            e = eager_ns[i]
            e_s = f"{e / 1e6:.6f}" if e is not None else ""
            table_rows.append(
                f"<tr><td>{label_esc}</td><td>{z}</td><td>{h_s}</td><td>{a_s}</td><td>{e_s}</td></tr>"
            )
        else:
            table_rows.append(
                f"<tr><td>{label_esc}</td><td>{z}</td><td>{h_s}</td><td>{a_s}</td></tr>"
            )

    thead = (
        "<tr><th>Payload</th><th>Size (bytes)</th><th>Host (ms)</th>"
        "<th>Accel (ms)</th><th>Accel eager (ms)</th></tr>"
        if include_eager
        else "<tr><th>Payload</th><th>Size (bytes)</th><th>Host (ms)</th><th>Accel (ms)</th></tr>"
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>fastdds_benchmark — latency sweep</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      line-height: 1.45;
      max-width: 960px;
      margin: 1rem auto;
      padding: 0 1rem;
      color: #1a1a1a;
    }}
    h1 {{ font-size: 1.35rem; }}
    h2 {{
      font-size: 1.1rem; margin-top: 1.5rem; border-bottom: 1px solid #ddd;
      padding-bottom: 0.25rem;
    }}
    .meta {{ list-style: none; padding: 0; margin: 0; }}
    .meta li {{ margin: 0.35rem 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.55rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f5f5f5; }}
    .plot img {{ max-width: 100%; height: auto; display: block; margin-top: 0.5rem; }}
    .files ul {{ margin: 0.25rem 0; }}
  </style>
</head>
<body>
  <h1>fastdds_benchmark — latency sweep</h1>
  <ul class="meta">
    <li><strong>Generated (local):</strong> {html.escape(generated_at_local, quote=True)}</li>
    <li><strong>Generated (UTC):</strong> {html.escape(generated_at_utc, quote=True)}</li>
    <li><strong>Runner:</strong> <code>{runner_s}</code></li>
    <li><strong>Samples per size / run:</strong> {samples}
        (<code>-w 1</code> → mean over indices ≥ 1)</li>
    <li><strong>Payload sizes:</strong> {len(sizes)} points from {sizes[0]} to
        {sizes[-1]} bytes</li>
    <li><strong>Binary:</strong> <code>{binary_s}</code></li>
    <li><strong>Accel path:</strong> {accel_s}</li>
    <li><strong>Accel eager sweep:</strong> {eager_on_s}</li>
  </ul>

  <h2>System</h2>
  <ul class="meta">
    {system_html}
  </ul>

  <h2>Plot</h2>
  <div class="plot">
    <img src="{plot_esc}" alt="Mean one-way latency vs payload size">
  </div>

  <h2>Mean latency (ms), indices ≥ 1</h2>
  <table>
    <thead>
      {thead}
    </thead>
    <tbody>
      {"".join(table_rows)}
    </tbody>
  </table>

  <h2>Files in this folder</h2>
  <div class="files">
    <ul>
      <li><code>{html.escape(plot_file, quote=True)}</code> — log–log chart</li>
      <li><code>{html.escape(csv_file, quote=True)}</code> — CSV
          (<code>size_bytes</code>, <code>flow</code>, <code>mean_ns</code>)</li>
      <li><code>{REPORT_MD}</code> — Markdown report</li>
      <li><code>{REPORT_HTML}</code> — this HTML report</li>
    </ul>
  </div>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def benchmark_binary_available(runner: Path, binary_arg: str | None) -> bool:
    """
    Fail fast when the benchmark executable is missing (default: sibling of
    run_latency_benchmark.sh). Skips check if -b is set to a valid executable, or if
    FASTDDS_BENCHMARK is set (resolved by the shell runner).
    """
    build_hint = (
        "Hint: build the Holoscan target first, e.g. "
        "cmake --build <your-build-dir> --target fastdds_benchmark, "
        "then run from the build output directory (binary next to run_latency_benchmark.sh), "
        "or pass -b PATH, or set FASTDDS_BENCHMARK."
    )
    if binary_arg:
        p = Path(binary_arg).expanduser()
        p = (Path.cwd() / p).resolve() if not p.is_absolute() else p.resolve()
        if p.is_file() and os.access(p, os.X_OK):
            return True
        print(
            f"Error: benchmark binary not found or not executable: {binary_arg}",
            file=sys.stderr,
        )
        print(build_hint, file=sys.stderr)
        return False
    if os.environ.get("FASTDDS_BENCHMARK", "").strip():
        return True
    sibling = runner.parent / "fastdds_benchmark"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return True
    print(
        f"Error: fastdds_benchmark not found next to runner ({sibling}).",
        file=sys.stderr,
    )
    print(build_hint, file=sys.stderr)
    return False


def parse_mean_ns(stdout: str) -> int | None:
    """Mean from the first stats block (`--- title (n=...) ---`) in runner output."""
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if not STATS_SECTION_HDR.match(line):
            continue
        for j in range(i + 1, len(lines)):
            inner = lines[j]
            if STATS_SECTION_HDR.match(inner):
                break
            m = MEAN_RE.search(inner)
            if m:
                return int(m.group(1).replace(",", ""))
    for m in MEAN_RE.finditer(stdout):
        return int(m.group(1).replace(",", ""))
    return None


def run_benchmark(
    runner: Path,
    *,
    samples: int,
    size_bytes: int,
    accel: bool,
    eager: bool = False,
    binary: str | None,
    extra_env: dict[str, str],
    timeout_sec: int | None,
) -> tuple[int | None, int]:
    if eager and not accel:
        raise ValueError("run_benchmark: eager=True requires accel=True")
    cmd = [
        "bash",
        str(runner),
        "-s",
        str(samples),
        "-z",
        str(size_bytes),
        "-w",
        "1",
    ]
    if accel:
        cmd.append("-a")
    if eager:
        cmd.append("-e")
    if binary:
        cmd.extend(["-b", binary])
    env = {**os.environ, **extra_env}
    timeout_val = timeout_sec if timeout_sec and timeout_sec > 0 else None
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_val,
        )
    except subprocess.TimeoutExpired as e:
        print(
            f"  [warn] size={size_bytes} accel={accel} eager={eager}: "
            f"timed out after {timeout_sec}s",
            file=sys.stderr,
        )
        if e.stdout:
            print(e.stdout[:2000], file=sys.stderr)
        if e.stderr:
            print(e.stderr[:2000], file=sys.stderr)
        return None, -1
    out = r.stdout + r.stderr
    if r.returncode != 0:
        print(
            f"  [warn] size={size_bytes} accel={accel} eager={eager} exit={r.returncode}",
            file=sys.stderr,
        )
        if r.stderr:
            print(r.stderr[:2000], file=sys.stderr)
        return None, r.returncode
    mean_ns = parse_mean_ns(out)
    if mean_ns is None:
        print(
            f"  [warn] size={size_bytes} accel={accel} eager={eager}: could not parse mean",
            file=sys.stderr,
        )
        return None, r.returncode
    return mean_ns, r.returncode


def main() -> int:
    here = Path(__file__).resolve().parent
    default_runner = here / "run_latency_benchmark.sh"

    p = argparse.ArgumentParser(
        description=(
            "Sweep fastdds_benchmark sizes and plot host vs accel vs optional accel+eager "
            "mean latency."
        ),
    )
    p.add_argument(
        "--runner",
        type=Path,
        default=default_runner,
        help="Path to run_latency_benchmark.sh (default: alongside this script)",
    )
    p.add_argument("-b", "--binary", help="Path to fastdds_benchmark (passed through to runner)")
    p.add_argument("-s", "--samples", type=int, default=20, help="Samples per run (default: 20)")
    p.add_argument(
        "--sizes",
        type=str,
        default="",
        help="Comma-separated payload sizes in bytes (default: built-in sweep)",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Report directory (default: ./latency_sweep_<YYYY-MM-DD_HHMMSS> "
            "using local time at start)"
        ),
    )
    p.add_argument(
        "--plot-name",
        default=PLOT_PNG,
        help=f"PNG filename inside the report directory (default: {PLOT_PNG})",
    )
    p.add_argument(
        "--csv-name",
        default=RESULTS_CSV,
        help=f"CSV filename inside the report directory (default: {RESULTS_CSV})",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write the CSV file in the report directory",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write REPORT.md / REPORT.html (still writes PNG and optional CSV)",
    )
    p.add_argument(
        "--no-accel",
        action="store_true",
        help="Only run host (plain Buffer) path",
    )
    p.add_argument(
        "--no-eager",
        action="store_true",
        help="Skip accel+eager runs (subscriber acquire_pointer_eager); no third plot line",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=0,
        metavar="SEC",
        help="Wall-clock limit per host/accel subprocess run (0 = none; avoids hung DDS in CI)",
    )
    args = p.parse_args()

    if plt is None or blended_transform_factory is None:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        return 1

    if args.sizes.strip():
        try:
            sizes = tuple(int(x.strip()) for x in args.sizes.split(",") if x.strip())
        except ValueError:
            print(
                f"Error: --sizes must be comma-separated integers (got {args.sizes!r}).",
                file=sys.stderr,
            )
            return 1
    else:
        sizes = DEFAULT_SIZES_BYTES

    if not sizes:
        print("No payload sizes to run (check --sizes).", file=sys.stderr)
        return 1

    for z in sizes:
        if z < 1:
            print(
                f"Error: payload sizes must be positive integers (got {z}).",
                file=sys.stderr,
            )
            return 1

    runner = args.runner.resolve()
    if not runner.is_file():
        print(f"Runner not found: {runner}", file=sys.stderr)
        print(
            "Hint: pass --runner PATH to run_latency_benchmark.sh (often copied next to the built "
            "binary after cmake --build … --target fastdds_benchmark).",
            file=sys.stderr,
        )
        return 1

    binary = args.binary
    if not benchmark_binary_available(runner, binary):
        return 1
    samples = args.samples
    if samples < 1:
        print("Error: --samples must be at least 1.", file=sys.stderr)
        return 1
    if samples < 3:
        print(
            "Error: need at least 3 samples for -w 1 (warmup excludes index 0; "
            "need ≥2 points for primary stdev).",
            file=sys.stderr,
        )
        return 1

    if args.timeout < 0:
        print("Error: --timeout must be >= 0.", file=sys.stderr)
        return 1
    timeout_sec = args.timeout if args.timeout > 0 else None

    report_dir = (
        args.output_dir if args.output_dir is not None else default_report_dir()
    ).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_path = report_dir / args.plot_name
    csv_path = report_dir / args.csv_name

    host_ns: list[int | None] = []
    accel_ns: list[int | None] = []
    eager_ns: list[int | None] = []
    include_eager = not args.no_accel and not args.no_eager

    print(f"Runner: {runner}")
    print(f"Report directory: {report_dir}")
    print(f"Samples per run: {samples} (warmup -w 1 → mean over indices >= 1)")
    print(f"Sizes ({len(sizes)}): {', '.join(str(s) for s in sizes)}")
    if timeout_sec:
        per = "host, accel, and accel+eager" if include_eager else "host and accel"
        print(f"Subprocess timeout: {timeout_sec}s per run ({per} each)")
    print()

    rows: list[dict[str, object]] = []

    for z in sizes:
        print(f"Size {z} bytes ({payload_label(z)}) …")
        h, _ = run_benchmark(
            runner,
            samples=samples,
            size_bytes=z,
            accel=False,
            eager=False,
            binary=binary,
            extra_env={},
            timeout_sec=timeout_sec,
        )
        host_ns.append(h)
        rows.append({"size_bytes": z, "flow": "host", "mean_ns": h if h is not None else ""})
        if not args.no_accel:
            a, _ = run_benchmark(
                runner,
                samples=samples,
                size_bytes=z,
                accel=True,
                eager=False,
                binary=binary,
                extra_env={},
                timeout_sec=timeout_sec,
            )
            accel_ns.append(a)
            rows.append({"size_bytes": z, "flow": "accel", "mean_ns": a if a is not None else ""})
            if include_eager:
                e, _ = run_benchmark(
                    runner,
                    samples=samples,
                    size_bytes=z,
                    accel=True,
                    eager=True,
                    binary=binary,
                    extra_env={},
                    timeout_sec=timeout_sec,
                )
                eager_ns.append(e)
                rows.append(
                    {
                        "size_bytes": z,
                        "flow": "accel_eager",
                        "mean_ns": e if e is not None else "",
                    }
                )
            else:
                eager_ns.append(None)
        else:
            accel_ns.append(None)
            eager_ns.append(None)
        h_s = f"{h / 1e6:.3f}" if h is not None else "n/a"
        a_s = f"{accel_ns[-1] / 1e6:.3f}" if accel_ns[-1] is not None else "n/a"
        e_s = f"{eager_ns[-1] / 1e6:.3f}" if eager_ns[-1] is not None else "n/a"
        if include_eager:
            print(f"  host mean: {h_s} ms | accel mean: {a_s} ms | accel eager: {e_s} ms\n")
        else:
            print(f"  host mean: {h_s} ms | accel mean: {a_s} ms\n")

    if not args.no_csv:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["size_bytes", "flow", "mean_ns"])
            w.writeheader()
            w.writerows(rows)

    x_kb = [s / 1024.0 for s in sizes]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    def series_ms(values: list[int | None]) -> list[float]:
        return [v / 1e6 if v is not None else float("nan") for v in values]

    ax.plot(
        x_kb,
        series_ms(host_ns),
        marker="o",
        linewidth=2,
        label="Host (plain Buffer)",
    )
    if not args.no_accel:
        ax.plot(
            x_kb,
            series_ms(accel_ns),
            marker="s",
            linewidth=2,
            label="Accel (AccelBuffer / CUDA IPC)",
        )
    if include_eager:
        ax.plot(
            x_kb,
            series_ms(eager_ns),
            marker="^",
            linewidth=2,
            label="Accel (eager acquire)",
        )

    ax.set_xlabel("Payload size (KiB)")
    ax.set_ylabel("Mean one-way latency (ms)")
    ax.set_title("fastdds_benchmark: subscriber − publisher (mean, indices ≥ 1)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()

    xaxis_top = blended_transform_factory(ax.transData, ax.transAxes)
    for sz, label in VIDEO_MARKERS:
        xk = sz / 1024.0
        if xk < min(x_kb) or xk > max(x_kb) * 1.01:
            continue
        ax.axvline(xk, color="0.75", linestyle="--", linewidth=1, zorder=0)
        ax.text(
            xk,
            0.99,
            label,
            transform=xaxis_top,
            rotation=90,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=8,
            color="0.35",
        )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)

    t_end = datetime.now(timezone.utc)
    if not args.no_report:
        local_tz = datetime.now().astimezone().tzinfo
        system_rows = collect_system_summary()
        report_kwargs = dict(
            generated_at_utc=t_end.strftime("%Y-%m-%d %H:%M:%S UTC"),
            generated_at_local=t_end.astimezone(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z"),
            runner=runner,
            samples=samples,
            sizes=sizes,
            binary=binary,
            no_accel=args.no_accel,
            plot_file=args.plot_name,
            csv_file=args.csv_name if not args.no_csv else "(not written)",
            host_ns=host_ns,
            accel_ns=accel_ns,
            eager_ns=eager_ns,
            include_eager=include_eager,
            system_rows=system_rows,
        )
        write_report_md(report_dir / REPORT_MD, **report_kwargs)
        write_report_html(report_dir / REPORT_HTML, **report_kwargs)

    print(f"Wrote report under {report_dir}:")
    print(f"  {plot_path.name}")
    if not args.no_csv:
        print(f"  {csv_path.name}")
    if not args.no_report:
        print(f"  {REPORT_MD}")
        print(f"  {REPORT_HTML}")

    if not any(h is not None for h in host_ns):
        print("Error: no successful host (plain Buffer) benchmark runs.", file=sys.stderr)
        return 1
    if not args.no_accel and not any(a is not None for a in accel_ns):
        print("Error: no successful AccelBuffer benchmark runs.", file=sys.stderr)
        return 1
    if include_eager and not any(e is not None for e in eager_ns):
        print("Error: no successful accel+eager benchmark runs.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
