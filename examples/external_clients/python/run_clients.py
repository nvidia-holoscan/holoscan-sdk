#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Python sensor and recorder clients against a C++ or Python relay graph."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import TextIO

INPUT_ENDPOINT = "example.external_clients.input"
OUTPUT_ENDPOINT = "example.external_clients.output"
SAMPLE_COUNT = 8
FIRST_VALUE = 7


def parse_args() -> argparse.Namespace:
    """Parse graph and client executable paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--sensor", required=True, type=Path)
    parser.add_argument("--recorder", required=True, type=Path)
    parser.add_argument("--async-recorder", action="store_true")
    return parser.parse_args()


def command(path: Path, *arguments: object) -> list[str]:
    """Invoke Python source with this interpreter and native binaries directly."""
    prefix = [sys.executable, str(path)] if path.suffix == ".py" else [str(path)]
    return [*prefix, *(str(argument) for argument in arguments)]


def stop_process(process: subprocess.Popen[str]) -> None:
    """Stop a child that did not reach its own terminal state."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def print_redirected_output(stdout: TextIO, stderr: TextIO) -> None:
    """Print one child's redirected streams in deterministic order."""
    for output in (stdout, stderr):
        output.flush()
        output.seek(0)
        print(output.read(), end="")


def print_captured_output(result: subprocess.CompletedProcess[str]) -> None:
    """Print a completed child's captured streams."""
    for output in (result.stdout, result.stderr):
        if output:
            print(output, end="")


def recorder_preserved_stream(output: str, first_value: int, count: int) -> bool:
    """Validate the exact finite payload sequence and its terminal EOS summary."""
    lines = output.splitlines()
    try:
        values = [
            int(line.split(maxsplit=1)[0].removeprefix("value="))
            for line in lines
            if line.startswith("value=")
        ]
    except (IndexError, ValueError):
        return False
    expected = list(range(first_value, first_value + count))
    return values == expected and lines[-1:] == [f"received={count} eos=remote"]


def main() -> int:
    """Run the three-process example and require clean EOS completion."""
    args = parse_args()
    instance = uuid.uuid4().hex
    input_endpoint = f"{INPUT_ENDPOINT}.{instance}"
    output_endpoint = f"{OUTPUT_ENDPOINT}.{instance}"
    with tempfile.TemporaryDirectory(prefix="holoscan-python-external-clients-") as temp_dir:
        temp_path = Path(temp_dir)
        catalog = temp_path / "endpoint-catalog.json"
        graph_stdout_path = temp_path / "graph.stdout"
        graph_stderr_path = temp_path / "graph.stderr"
        recorder_stdout_path = temp_path / "recorder.stdout"
        recorder_stderr_path = temp_path / "recorder.stderr"
        with (
            graph_stdout_path.open("w+", encoding="utf-8") as graph_stdout,
            graph_stderr_path.open("w+", encoding="utf-8") as graph_stderr,
            recorder_stdout_path.open("w+", encoding="utf-8") as recorder_stdout,
            recorder_stderr_path.open("w+", encoding="utf-8") as recorder_stderr,
        ):
            graph = subprocess.Popen(
                command(args.graph, "--catalog", catalog, "--instance", instance),
                stdout=graph_stdout,
                stderr=graph_stderr,
                text=True,
            )
            recorder: subprocess.Popen[str] | None = None
            try:
                deadline = time.monotonic() + 10.0
                while not catalog.exists() and graph.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not catalog.exists():
                    print("graph did not publish its endpoint catalog", file=sys.stderr)
                    return 1

                recorder_arguments: list[object] = [catalog, output_endpoint]
                if args.async_recorder:
                    recorder_arguments.append("--async")
                recorder = subprocess.Popen(
                    command(args.recorder, *recorder_arguments),
                    stdout=recorder_stdout,
                    stderr=recorder_stderr,
                    text=True,
                )
                deadline = time.monotonic() + 10.0
                recorder_ready = False
                while time.monotonic() < deadline:
                    if "subscriber ready" in recorder_stdout_path.read_text(encoding="utf-8"):
                        recorder_ready = True
                        break
                    if recorder.poll() is not None or graph.poll() is not None:
                        break
                    time.sleep(0.01)
                if not recorder_ready:
                    # A short fixed-count recorder can exit between poll and observation. Drain its
                    # final flushed output before deciding that attachment never completed.
                    recorder_ready = "subscriber ready" in recorder_stdout_path.read_text(
                        encoding="utf-8"
                    )
                if not recorder_ready:
                    print("recorder did not attach to the graph export", file=sys.stderr)
                    return 1
                sensor = subprocess.run(
                    command(args.sensor, catalog, input_endpoint, SAMPLE_COUNT, 1, FIRST_VALUE),
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
                graph.wait(timeout=25)
                recorder.wait(timeout=25)

                print_redirected_output(graph_stdout, graph_stderr)
                print_redirected_output(recorder_stdout, recorder_stderr)
                print_captured_output(sensor)
                if graph.returncode != 0 or recorder.returncode != 0 or sensor.returncode != 0:
                    return graph.returncode or recorder.returncode or sensor.returncode
                recorder_output = recorder_stdout_path.read_text(encoding="utf-8")
                if not recorder_preserved_stream(recorder_output, FIRST_VALUE, SAMPLE_COUNT):
                    print("recorder did not preserve the expected stream", file=sys.stderr)
                    return 1
                if args.async_recorder and ("asyncio helper threads=0" not in recorder_output):
                    print("async recorder did not verify its thread count", file=sys.stderr)
                    return 1
            except (OSError, subprocess.TimeoutExpired) as error:
                print(f"external client round trip failed: {error}", file=sys.stderr)
                return 1
            finally:
                if recorder is not None:
                    stop_process(recorder)
                stop_process(graph)

    print("Python external clients round trip complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
