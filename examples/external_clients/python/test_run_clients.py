# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the standalone external-client process harness."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest
import run_clients


def recorder_output(values: list[int], *, asyncio_marker: bool = False) -> str:
    """Build the recorder's structured value lines and terminal summary."""
    lines = [f"value={value} sequence={index}" for index, value in enumerate(values)]
    if asyncio_marker:
        lines.append("asyncio helper threads=0")
    lines.append(f"received={len(values)} eos=remote")
    return "\n".join(lines) + "\n"


def run_harness(monkeypatch, output: str, *, async_recorder: bool = False) -> int:
    """Run the real harness control flow against deterministic child-process seams."""
    arguments = argparse.Namespace(
        graph=Path("relay"),
        sensor=Path("sensor.py"),
        recorder=Path("recorder.py"),
        async_recorder=async_recorder,
    )
    monkeypatch.setattr(run_clients, "parse_args", lambda: arguments)

    class Process:
        def __init__(self, command, *, stdout, stderr, text):
            del stderr, text
            self.returncode = None
            if "--catalog" in command:
                catalog_index = command.index("--catalog") + 1
                Path(command[catalog_index]).write_text("{}", encoding="utf-8")
            else:
                stdout.write("subscriber ready\n")
                stdout.write(output)
                stdout.flush()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            self.returncode = 0
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(run_clients.subprocess, "Popen", Process)

    def completed_run(command, *args, **kwargs):
        del args, kwargs
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        run_clients.subprocess,
        "run",
        completed_run,
    )
    return run_clients.main()


@pytest.mark.parametrize("async_recorder", [False, True])
def test_harness_accepts_exact_ordered_stream(monkeypatch, async_recorder):
    values = list(
        range(run_clients.FIRST_VALUE, run_clients.FIRST_VALUE + run_clients.SAMPLE_COUNT)
    )

    assert (
        run_harness(
            monkeypatch,
            recorder_output(values, asyncio_marker=async_recorder),
            async_recorder=async_recorder,
        )
        == 0
    )


@pytest.mark.parametrize(
    "values",
    [
        [],
        [99] * run_clients.SAMPLE_COUNT,
        list(range(run_clients.FIRST_VALUE, run_clients.FIRST_VALUE + run_clients.SAMPLE_COUNT))[
            ::-1
        ],
        list(
            range(run_clients.FIRST_VALUE, run_clients.FIRST_VALUE + run_clients.SAMPLE_COUNT - 1)
        ),
    ],
    ids=["empty", "wrong-values", "reordered", "truncated"],
)
def test_harness_rejects_corrupted_streams_after_eos(monkeypatch, values):
    assert run_harness(monkeypatch, recorder_output(values)) == 1
