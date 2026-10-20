#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the standard matched Channel domain/transport comparison matrix."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_channel_benchmark.py"
SUMMARIZER = SCRIPT_DIR / "summarize_channel_benchmark_results.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_binary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--domains",
        default="interprocess",
        help=(
            "Comma-separated execution modes: intraprocess, intraprocess_sidecar, "
            "interprocess. Sidecar mode runs only Tensor and Image rows."
        ),
    )
    parser.add_argument("--pod-samples", type=int, default=100_000)
    parser.add_argument("--host-samples", type=int, default=20_000)
    parser.add_argument("--device-samples", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--tensor-bytes", type=int, default=1_048_576)
    parser.add_argument("--workload", choices=("throughput", "latency"), default="throughput")
    parser.add_argument("--subscriber-queue-depth", type=int, default=64)
    parser.add_argument("--publisher-period-us", type=int, default=0)
    parser.add_argument("--publisher-pacing", choices=("sleep", "spin"), default="sleep")
    parser.add_argument("--payload-validation", choices=("none", "final", "all"), default="final")
    parser.add_argument("--publisher-cpu", type=int, default=-1)
    parser.add_argument("--subscriber-cpu", type=int, default=-1)
    parser.add_argument(
        "--subscriber-cpus",
        default=None,
        help="Comma-separated subscriber CPU list forwarded to fanout rows.",
    )
    parser.add_argument(
        "--fanouts",
        default="1",
        help="Comma-separated subscriber counts for POD, Tensor host/device, and Image rows.",
    )
    parser.add_argument(
        "--payloads",
        default="pod,tensor,image",
        help="Comma-separated payload families: pod, tensor, and/or image.",
    )
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    return parser.parse_args()


def byte_suffix(num_bytes: int) -> str:
    if num_bytes % (1024 * 1024) == 0:
        return f"{num_bytes // (1024 * 1024)}m"
    if num_bytes % 1024 == 0:
        return f"{num_bytes // 1024}k"
    return f"{num_bytes}b"


def parse_fanouts(text: str) -> list[int]:
    fanouts = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not fanouts or any(fanout <= 0 for fanout in fanouts):
        raise ValueError("--fanouts must contain positive subscriber counts")
    return list(dict.fromkeys(fanouts))


def parse_domains(text: str) -> list[str]:
    domains = [item.strip() for item in text.split(",") if item.strip()]
    valid_domains = {"intraprocess", "intraprocess_sidecar", "interprocess"}
    if not domains or any(domain not in valid_domains for domain in domains):
        raise ValueError(
            "--domains must contain intraprocess, intraprocess_sidecar, and/or interprocess"
        )
    return list(dict.fromkeys(domains))


def parse_payloads(text: str) -> list[str]:
    payloads = [item.strip() for item in text.split(",") if item.strip()]
    valid_payloads = {"pod", "tensor", "image"}
    if not payloads or any(payload not in valid_payloads for payload in payloads):
        raise ValueError("--payloads must contain pod, tensor, and/or image")
    return list(dict.fromkeys(payloads))


def row_name(base: str, subscribers: int) -> str:
    return base if subscribers == 1 else f"{base}_1to{subscribers}"


def matrix_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    suffix = byte_suffix(args.tensor_bytes)
    fanouts = parse_fanouts(args.fanouts)
    payloads = parse_payloads(args.payloads)
    rows: list[dict[str, object]] = []

    if "pod" in payloads:
        rows.extend(
            {
                "name": row_name("pod", subscribers),
                "payload": "pod",
                # The runner requires a Tensor storage option even though POD transport is inline.
                "storage": "pinned_host",
                "samples": args.pod_samples,
                "subscribers": subscribers,
            }
            for subscribers in fanouts
        )

    if "tensor" in payloads:
        rows.append(
            {
                "name": f"tensor_pageable_host_{suffix}",
                "payload": "tensor",
                "storage": "pageable_host",
                "samples": args.host_samples,
                "subscribers": 1,
            }
        )
        for storage, samples in (
            ("pinned_host", args.host_samples),
            ("device", args.device_samples),
        ):
            base = f"tensor_{storage}_{suffix}"
            rows.extend(
                {
                    "name": row_name(base, subscribers),
                    "payload": "tensor",
                    "storage": storage,
                    "samples": samples,
                    "subscribers": subscribers,
                }
                for subscribers in fanouts
            )

    if "image" in payloads:
        for storage, samples in (
            ("pageable_host", args.host_samples),
            ("pinned_host", args.host_samples),
            ("device", args.device_samples),
        ):
            base = f"image_{storage}_{suffix}"
            rows.extend(
                {
                    "name": row_name(base, subscribers),
                    "payload": "image",
                    "storage": storage,
                    "samples": samples,
                    "subscribers": subscribers,
                }
                for subscribers in fanouts
            )
    return [
        {**row, "domain": domain, "name": f"{domain}_{row['name']}"}
        for domain in parse_domains(args.domains)
        for row in rows
        if domain != "intraprocess_sidecar" or row["payload"] != "pod"
    ]


def run_command(command: list[str], stdout_path: Path | None = None) -> None:
    if stdout_path is None:
        subprocess.run(command, check=True)
        return
    with stdout_path.open("w", encoding="utf-8") as stdout:
        subprocess.run(command, check=True, stdout=stdout, text=True)


def append_command(
    commands_path: Path, command: list[str], stdout_path: Path | None = None
) -> None:
    with commands_path.open("a", encoding="utf-8") as commands:
        commands.write(shlex.join(command))
        if stdout_path is not None:
            commands.write(f" > {shlex.quote(str(stdout_path))}")
        commands.write("\n")


def main() -> int:
    args = parse_args()
    if args.workload == "latency" and args.publisher_period_us <= 0:
        raise ValueError("latency workload requires a nonzero --publisher-period-us")
    if args.workload == "latency" and args.subscriber_queue_depth not in {1, 2}:
        raise ValueError("latency workload requires --subscriber-queue-depth 1 or 2")
    if args.workload == "throughput" and args.publisher_period_us != 0:
        raise ValueError("--publisher-period-us requires --workload latency")
    if args.workload != "latency" and args.publisher_pacing == "spin":
        raise ValueError("--publisher-pacing spin requires --workload latency")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(f"channel-comparison-matrix-{time.strftime('%Y%m%d-%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    commands_path = output_dir / "commands.sh"
    commands_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n", encoding="utf-8")

    for row in matrix_rows(args):
        row_name = str(row["name"])
        row_dir = output_dir / row_name
        row_json = output_dir / f"{row_name}.json"
        command = [
            sys.executable,
            str(RUNNER),
            str(args.benchmark_binary),
            "--domain",
            str(row["domain"]),
            "--payload",
            str(row["payload"]),
            "--tensor-storage",
            str(row["storage"]),
            "--tensor-bytes",
            str(args.tensor_bytes),
            "--workload",
            args.workload,
            "--samples",
            str(row["samples"]),
            "--warmup",
            str(args.warmup),
            "--subscriber-queue-depth",
            str(args.subscriber_queue_depth),
            "--publisher-period-us",
            str(args.publisher_period_us),
            "--publisher-pacing",
            args.publisher_pacing,
            "--payload-validation",
            args.payload_validation,
            "--publisher-cpu",
            str(args.publisher_cpu),
            "--subscriber-cpu",
            str(args.subscriber_cpu),
            "--expected-subscribers",
            str(row["subscribers"]),
            "--timeout-sec",
            str(args.timeout_sec),
            "--output-dir",
            str(row_dir),
        ]
        if args.subscriber_cpus is not None:
            command.extend(["--subscriber-cpus", args.subscriber_cpus])
        append_command(commands_path, command, stdout_path=row_json)
        print(f"running {row_name} ...", flush=True)
        run_command(command, stdout_path=row_json)

    summary_command = [sys.executable, str(SUMMARIZER), str(output_dir)]
    append_command(commands_path, summary_command, stdout_path=output_dir / "summary.md")
    with (output_dir / "summary.md").open("w", encoding="utf-8") as summary:
        subprocess.run(summary_command, check=True, stdout=summary, text=True)
    print((output_dir / "summary.md").read_text(encoding="utf-8"), end="")
    print(f"\nresults: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
