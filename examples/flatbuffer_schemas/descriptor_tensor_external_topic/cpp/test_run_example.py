# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO, cast

import pytest
import run_example as runner


class FakeGraphProcess:
    def __init__(self, *, wait_times_out: bool = False) -> None:
        self.returncode: int | None = None
        self.wait_times_out = wait_times_out
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.wait_times_out and timeout == 25:
            raise subprocess.TimeoutExpired("graph", timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def configure_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    args = SimpleNamespace(graph=tmp_path / "graph", client=tmp_path / "client")
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    return args


def install_graph_process(
    monkeypatch: pytest.MonkeyPatch,
    graph: FakeGraphProcess,
    *,
    create_catalog: bool = True,
) -> None:
    def start_graph(command: list[str], **kwargs: object) -> FakeGraphProcess:
        catalog = Path(command[command.index("--catalog") + 1])
        if create_catalog:
            catalog.touch()
        cast(TextIO, kwargs["stdout"]).write("graph stdout\n")
        cast(TextIO, kwargs["stderr"]).write("graph stderr\n")
        return graph

    monkeypatch.setattr(runner.subprocess, "Popen", start_graph)


def test_graph_startup_oserror_is_controlled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = configure_arguments(monkeypatch, tmp_path)

    # Key setup: fail before Popen can assign the local graph process.
    def fail_start(*_args: object, **_kwargs: object) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(runner.subprocess, "Popen", fail_start)

    # Key assertion: failure before graph assignment returns cleanly without entering cleanup.
    assert runner.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"failed to start graph executable '{args.graph}': permission denied" in captured.err


def test_client_startup_oserror_stops_graph_and_prints_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = configure_arguments(monkeypatch, tmp_path)
    graph = FakeGraphProcess()
    install_graph_process(monkeypatch, graph)

    # Key setup: fail the standalone peer launch after the graph has started.
    def fail_start(*_args: object, **_kwargs: object) -> None:
        raise OSError("exec format error")

    monkeypatch.setattr(runner.subprocess, "run", fail_start)

    # Key assertion: the controlled failure stops the graph and retains both diagnostics.
    assert runner.main() == 1
    captured = capsys.readouterr()
    assert graph.terminated
    assert "graph stdout" in captured.out
    assert "graph stderr" in captured.out
    assert f"failed to start client executable '{args.client}': exec format error" in captured.err


def test_graph_wait_timeout_stops_graph_and_prints_both_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = configure_arguments(monkeypatch, tmp_path)
    graph = FakeGraphProcess(wait_times_out=True)
    install_graph_process(monkeypatch, graph)
    # Key setup: let the standalone peer finish while the graph remains live.
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="client stdout\n", stderr="client stderr\n"
        ),
    )

    # Key assertion: the graph timeout stops the process and emits both process logs.
    assert runner.main() == 1
    captured = capsys.readouterr()
    assert graph.terminated
    assert "graph stdout" in captured.out
    assert "graph stderr" in captured.out
    assert "client stdout" in captured.out
    assert "client stderr" in captured.out
    assert f"graph executable '{args.graph}' timed out after 25 seconds" in captured.err


def test_client_timeout_stops_graph_and_prints_partial_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = configure_arguments(monkeypatch, tmp_path)
    graph = FakeGraphProcess()
    install_graph_process(monkeypatch, graph)

    # Key setup: retain partial peer output in the timeout exception.
    def time_out(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(
            cmd="client",
            timeout=20,
            output="client partial stdout\n",
            stderr="client partial stderr\n",
        )

    monkeypatch.setattr(runner.subprocess, "run", time_out)

    # Key assertion: timeout cleanup emits graph output before the peer's partial output.
    assert runner.main() == 1
    captured = capsys.readouterr()
    assert graph.terminated
    assert "graph stdout" in captured.out
    assert "graph stderr" in captured.out
    assert "client partial stdout" in captured.out
    assert "client partial stderr" in captured.out
    assert captured.out.index("graph stdout") < captured.out.index("client partial stdout")
    assert f"client executable '{args.client}' timed out after 20 seconds" in captured.err


def test_missing_catalog_returns_one_after_stopping_graph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_arguments(monkeypatch, tmp_path)
    graph = FakeGraphProcess()
    install_graph_process(monkeypatch, graph, create_catalog=False)
    monotonic_values = iter((0.0, 11.0))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic_values))

    # Key assertion: SIGTERM's negative return code does not escape as a shell exit status.
    assert runner.main() == 1
    captured = capsys.readouterr()
    assert graph.terminated
    assert graph.returncode == -15
    assert "graph stdout" in captured.out
    assert "graph stderr" in captured.out
