#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the descriptor-Tensor external-topic graph and standalone peer."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--client", required=True, type=Path)
    return parser.parse_args()


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def print_captured_output(stdout: str | bytes | None, stderr: str | bytes | None) -> None:
    for output in (stdout, stderr):
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        if output:
            print(output, end="")


def print_redirected_output(stdout: TextIO, stderr: TextIO) -> None:
    for output in (stdout, stderr):
        output.flush()
        output.seek(0)
        print(output.read(), end="")


def command(path: Path, *arguments: object) -> list[str]:
    """Invoke Python source with this interpreter and native binaries directly."""
    prefix = [sys.executable, str(path)] if path.suffix == ".py" else [str(path)]
    return [*prefix, *(str(argument) for argument in arguments)]


def main() -> int:
    args = parse_args()
    instance = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="holoscan-descriptor-topic-") as temp_dir:
        catalog = Path(temp_dir) / "endpoint-catalog.json"
        graph_stdout_path = Path(temp_dir) / "graph.stdout"
        graph_stderr_path = Path(temp_dir) / "graph.stderr"
        with (
            graph_stdout_path.open("w+", encoding="utf-8") as graph_stdout_file,
            graph_stderr_path.open("w+", encoding="utf-8") as graph_stderr_file,
        ):
            try:
                graph = subprocess.Popen(
                    command(args.graph, "--catalog", catalog, "--instance", instance),
                    stdout=graph_stdout_file,
                    stderr=graph_stderr_file,
                    text=True,
                )
            except OSError as error:
                print(f"failed to start graph executable '{args.graph}': {error}", file=sys.stderr)
                return 1
            try:
                deadline = time.monotonic() + 10.0
                while not catalog.exists() and graph.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not catalog.exists():
                    stop_process(graph)
                    print_redirected_output(graph_stdout_file, graph_stderr_file)
                    return 1

                try:
                    client = subprocess.run(
                        command(args.client, "--catalog", catalog, "--instance", instance),
                        capture_output=True,
                        text=True,
                        timeout=20,
                        check=False,
                    )
                except OSError as error:
                    print(
                        f"failed to start client executable '{args.client}': {error}",
                        file=sys.stderr,
                    )
                    stop_process(graph)
                    print_redirected_output(graph_stdout_file, graph_stderr_file)
                    return 1
                except subprocess.TimeoutExpired as error:
                    print(
                        f"client executable '{args.client}' timed out after 20 seconds",
                        file=sys.stderr,
                    )
                    stop_process(graph)
                    print_redirected_output(graph_stdout_file, graph_stderr_file)
                    print_captured_output(error.stdout, error.stderr)
                    return 1
                try:
                    graph.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    print(
                        f"graph executable '{args.graph}' timed out after 25 seconds",
                        file=sys.stderr,
                    )
                    stop_process(graph)
                    print_redirected_output(graph_stdout_file, graph_stderr_file)
                    print_captured_output(client.stdout, client.stderr)
                    return 1
                print_redirected_output(graph_stdout_file, graph_stderr_file)
                print_captured_output(client.stdout, client.stderr)
                if graph.returncode != 0 or client.returncode != 0:
                    return graph.returncode or client.returncode
            finally:
                stop_process(graph)

    print("descriptor Tensor external-topic round trip complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
