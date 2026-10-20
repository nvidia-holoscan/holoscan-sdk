# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions; the display smoke runs directly through CTest, never here."""

import datetime
import inspect
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import holoviz_image_topic_lifecycle as lifecycle
import pytest

example = lifecycle.example


class Timeline:
    def __init__(self, signal_at=None):
        self.now = 0.0
        self.signal_at = signal_at
        self.waits = []

    def is_set(self):
        return self.signal_at is not None and self.now >= self.signal_at

    def wait(self, timeout):
        self.waits.append(timeout)
        self.now += timeout
        return self.is_set()


class Session:
    def __init__(self, timeline, terminal_at=None, diagnostics=(), failed=False):
        self.timeline = timeline
        self.terminal_at = terminal_at
        self.diagnostics = diagnostics
        self.termination = (
            example.hs.RunTermination.FAILED if failed else example.hs.RunTermination.STOPPED
        )
        self.run_waits = []
        self.stopped = False

    def wait(self, timeout=None):
        if self.stopped:
            return True
        if timeout:
            self.run_waits.append((self.timeline.now, timeout))
            self.timeline.now += timeout
        return self.terminal_at is not None and self.timeline.now >= self.terminal_at

    def request_stop(self):
        self.stopped = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def run(timeline, session, seconds=1.0, startup_timeout=0.2):
    return lifecycle._run_lifecycle(
        session, timeline, seconds, startup_timeout, monotonic=lambda: timeline.now
    )


def diagnostic(
    message,
    token="RUNTIME_LIFECYCLE_CALLBACK_FAILED",
    stage=None,
    operator_path="visualizer",
    code=example.hs.errors.ErrorCode.NOT_FOUND,
):
    return SimpleNamespace(
        token=token,
        detail=SimpleNamespace(
            operator_path=operator_path,
            stage=stage or example.hs.lifecycle.LifecycleStage.START,
            error=SimpleNamespace(code=code, message=message),
        ),
    )


# REQ: PY-API-019
def test_tutorial_graph_needs_no_startup_witness_argument():
    assert not inspect.signature(example.build_graph).parameters


# REQ: PY-API-019
# REQ: LIFECYCLE-001
def test_driver_composes_actual_example_graph_on_existing_clock():
    graph = example.build_graph()
    tutorial_plan = example.hs.compile(graph)
    assert tutorial_plan, tutorial_plan.diagnostics
    tutorial = json.loads(tutorial_plan.json())
    assert [row["operator_path"] for row in tutorial["operator_parameters"]] == ["visualizer"]
    assert not tutorial["clock_triggers"]
    lifecycle.add_startup_observer(graph, threading.Event())
    observed_plan = example.hs.compile(graph)
    assert observed_plan, observed_plan.diagnostics
    observed = json.loads(observed_plan.json())
    assert observed["clock_catalog"]["clocks"] == tutorial["clock_catalog"]["clocks"]
    assert not observed["lifecycle_dependencies"]
    assert {row["operator_path"] for row in observed["operator_parameters"]} == {
        "visualizer",
        "test_startup_observer",
    }
    (trigger,) = observed["clock_triggers"]
    assert trigger["clock"] == tutorial["clock_catalog"]["root_default_clock"]
    assert trigger["global_start_barrier"]


# REQ: PY-API-019
@pytest.mark.parametrize(
    "case,expected",
    [
        ("clean", 0),
        ("early", 1),
        ("display", 77),
        ("renderer", 1),
        ("failed", 1),
        ("shutdown_timeout", 1),
    ],
)
def test_tutorial_session_result(case, expected, monkeypatch, capsys):
    session = Session(
        Timeline(), terminal_at=0 if case == "early" else None, failed=case == "failed"
    )
    if case in ("display", "renderer"):
        session.diagnostics = [
            diagnostic(
                "create_window failed: [holoviz] No window system route"
                if case == "display"
                else "renderer failed"
            )
        ]
    if case == "shutdown_timeout":
        session.wait = lambda **_kwargs: False
    monkeypatch.setattr(sys, "argv", ["holoviz_image_topic.py", "--seconds=1"])
    monkeypatch.setattr(example, "build_graph", lambda: None)
    monkeypatch.setattr(example.hs, "compile", lambda _: object())
    monkeypatch.setattr(example.hs, "run_async", lambda _: session)
    assert example.main() == expected
    assert session.stopped
    output = capsys.readouterr()
    if case == "clean":
        assert "session wait completed" in output.out
        assert session.run_waits == [(0.0, 1.0)]
    if case == "shutdown_timeout":
        assert "shutdown timeout" in output.err


# REQ: PY-API-019
@pytest.mark.parametrize(
    "entry,option",
    [(example, "--seconds"), (lifecycle, "--seconds"), (lifecycle, "--startup-timeout")],
)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_invalid_arguments_never_start(entry, option, value, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["holoviz_image_topic.py", f"{option}={value}"])
    monkeypatch.setattr(example.hs, "run_async", lambda *_args: pytest.fail("session constructed"))
    monkeypatch.setattr(example.hs, "compile", lambda *_args: pytest.fail("graph compiled"))
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
    assert f"invalid {option} value" in capsys.readouterr().err


# REQ: PY-API-019
def test_invalid_argument_process_exit():
    option, value = "--seconds", "0"
    result = subprocess.run(
        [sys.executable, str(Path(example.__file__)), f"{option}={value}"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert f"invalid {option} value" in result.stderr


# REQ: PY-API-019
@pytest.mark.parametrize("signal_at", [None, 0.21])
def test_pending_start_fails_before_cleanup(signal_at, capsys):
    timeline = Timeline(signal_at)
    session = Session(timeline)
    original_stop = session.request_stop

    def stop():
        assert "startup timeout" in capsys.readouterr().err
        original_stop()

    session.request_stop = stop
    assert run(timeline, session) == 1
    assert session.stopped
    assert not session.run_waits
    assert timeline.now == pytest.approx(0.2)
    assert all(0 < wait <= 0.05 for wait in timeline.waits)


# REQ: PY-API-019
@pytest.mark.parametrize("signal_at", [0, 0.15, 0.2])
def test_full_duration_follows_start(signal_at):
    timeline = Timeline(signal_at)
    session = Session(timeline)
    assert run(timeline, session) == 0
    assert session.run_waits == [(pytest.approx(signal_at), 1.0)]
    assert session.stopped


# REQ: PY-API-019
@pytest.mark.parametrize("terminal_at", [0, 0.1, 0.5])
def test_early_terminal_cannot_pass(terminal_at):
    timeline = Timeline(0.1)
    assert run(timeline, Session(timeline, terminal_at=terminal_at)) == 1


# REQ: PY-API-019
def test_event_terminal_race_uses_second_probe():
    timeline = Timeline(0)
    session = Session(timeline)
    probes = iter([False, True])
    session.wait = lambda **_kwargs: True if session.stopped else next(probes)
    assert run(timeline, session) == 1
    assert not session.run_waits


# REQ: PY-API-019
@pytest.mark.parametrize(
    "diagnostics,failed", [([diagnostic("renderer failed")], False), ([], True)]
)
def test_final_failure_overrides_start(diagnostics, failed):
    timeline = Timeline(0)
    session = Session(timeline, diagnostics=diagnostics, failed=failed)
    assert run(timeline, session) == 1


# REQ: PY-API-019
@pytest.mark.parametrize(
    "diagnostics,expected",
    [
        ([], False),
        ([diagnostic("create_window failed")], False),
        ([diagnostic("Failed to initialize GLFW")], False),
        ([diagnostic("Failed to create GLFW window")], False),
        ([diagnostic("create_window failed: [holoviz] No window system route")], True),
        (
            [
                diagnostic(
                    "[holoviz] No window system route", code=example.hs.errors.ErrorCode.FAILURE
                )
            ],
            False,
        ),
        (
            [
                diagnostic(
                    "create_window failed: [holoviz] No window system route", operator_path="other"
                )
            ],
            False,
        ),
        ([diagnostic("CUDA failed")], False),
        ([diagnostic("create_window failed"), diagnostic("renderer failed")], False),
        ([diagnostic("create_window failed", token="OTHER")], False),
        (
            [diagnostic("create_window failed", stage=example.hs.lifecycle.LifecycleStage.STOP)],
            False,
        ),
        ([SimpleNamespace(token="RUNTIME_LIFECYCLE_CALLBACK_FAILED", detail=None)], False),
    ],
)
def test_display_classifier(diagnostics, expected):
    assert example.display_unavailable(SimpleNamespace(diagnostics=diagnostics)) is expected


# REQ: PY-API-019
def test_only_display_failure_skips():
    timeline = Timeline()
    session = Session(
        timeline,
        terminal_at=0,
        diagnostics=[diagnostic("create_window failed: [holoviz] No window system route")],
    )
    assert run(timeline, session) == 77
    session.diagnostics.append(diagnostic("CUDA failed"))
    assert run(Timeline(), session) == 1


# REQ: PY-API-019
# REQ: PY-OP-001
# REQ: LIFECYCLE-001
# REQ: LIFECYCLE-004
@pytest.mark.parametrize(
    "case", ["python_delay", "python_fail", "cpp_delay", "cpp_fail", "global_fail", "isolation"]
)
def test_hosted_start_in_subprocess(case):
    result = subprocess.run(
        [sys.executable, str(Path(__file__)), case],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def wait_for_readiness(messages, marker, timeout, *, monotonic=time.monotonic):
    """Require the callback marker within a setup budget independent of the hang guard."""
    deadline = monotonic() + timeout
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise AssertionError(f"setup failed: missing readiness marker {marker}")
        try:
            line = messages.get(timeout=remaining)
        except queue.Empty as error:
            raise AssertionError(f"setup failed: missing readiness marker {marker}") from error
        if line is None:
            raise AssertionError(f"setup failed: child exited before readiness marker {marker}")
        if line.strip() == marker:
            if monotonic() > deadline:
                raise AssertionError(f"setup failed: late readiness marker {marker}")
            return


def run_watchdog_child(command, marker, hang_timeout):
    """Drain output, wait for callback entry, then enforce and reap the expected hang."""
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as process:
        messages = queue.Queue()
        output = []

        def read_output():
            for line in process.stdout:
                output.append(line)
                messages.put(line)
            messages.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            wait_for_readiness(messages, marker, 30.0)
            try:
                process.wait(timeout=hang_timeout)
            except subprocess.TimeoutExpired:
                pass
            else:
                raise AssertionError(f"callback unexpectedly returned: {process.returncode}")
        except AssertionError as error:
            raise AssertionError(f"{error}\n{''.join(output)}") from error
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            reader.join(timeout=5)
        assert not reader.is_alive(), "child output reader did not finish"
        return "".join(output)


# REQ: PY-API-019
@pytest.mark.parametrize(
    "case,timeout,marker,message",
    [
        ("hung_start", 2, "START_CALLBACK_ENTERED", "startup timeout"),
        ("hung_cleanup", 13, "STOP_CALLBACK_ENTERED", "shutdown timeout"),
    ],
)
def test_outer_watchdog_bounds_stuck_callbacks(case, timeout, marker, message):
    output = run_watchdog_child([sys.executable, str(Path(__file__)), case], marker, timeout)
    assert message in output


# REQ: PY-API-019
@pytest.mark.parametrize("case", ["missing", "exited", "late", "ready"])
def test_watchdog_readiness_deadline(case):
    now = [0.0]

    class Messages:
        def get(self, timeout):
            assert timeout == 5.0
            if case == "missing":
                raise queue.Empty
            if case == "exited":
                return None
            now[0] = 5.1 if case == "late" else 4.0
            return "CALLBACK_ENTERED\n"

    if case == "ready":
        wait_for_readiness(Messages(), "CALLBACK_ENTERED", 5.0, monotonic=lambda: now[0])
    else:
        with pytest.raises(AssertionError, match="setup failed: .*readiness marker"):
            wait_for_readiness(Messages(), "CALLBACK_ENTERED", 5.0, monotonic=lambda: now[0])


# REQ: PY-API-019
@pytest.mark.parametrize("ready", [False, True])
def test_watchdog_child_exit_is_a_failure(ready):
    command = [sys.executable, "-c", "print('CALLBACK_ENTERED', flush=True)" if ready else "pass"]
    message = "callback unexpectedly returned" if ready else "setup failed: child exited"
    with pytest.raises(AssertionError, match=message):
        run_watchdog_child(command, "CALLBACK_ENTERED", 1.0)


def hosted_case(case):
    """Exercise global START, real hosted reconstruction and run-local observation."""
    import holoscan_test_support  # noqa: F401, PLC0415  # The fixture is never installed.

    hs = example.hs
    started = threading.Event()
    entered = threading.Event()
    release = threading.Event()

    class Predecessor(hs.Operator):
        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=datetime.timedelta(seconds=1)))

        def start(self):
            if case == "hung_start":
                # Deterministically exceed the 100ms observation guard before callback entry.
                time.sleep(0.25)
            entered.set()
            if case == "hung_start":
                print("START_CALLBACK_ENTERED", flush=True)
                threading.Event().wait()
            assert release.wait(5)
            if case == "python_fail":
                raise RuntimeError("intentional predecessor START failure")

        def stop(self):
            if case == "hung_cleanup":
                print("STOP_CALLBACK_ENTERED", flush=True)
                threading.Event().wait()

        def compute(self, context):
            pass

    graph = hs.Graph("startup-observer-test")
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    if case.startswith("cpp"):
        graph.op("holoscan.test.StartupProbe", "predecessor", delay_ms=250, fail=case == "cpp_fail")
    else:
        graph.op(Predecessor, "predecessor")
    lifecycle.add_startup_observer(graph, started)

    class FailingStart(hs.Operator):
        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=datetime.timedelta(seconds=1)))

        def start(self):
            raise RuntimeError("intentional failure in another START callback")

        def compute(self, context):
            raise AssertionError("compute cannot precede global START completion")

    if case == "global_fail":
        graph.op(FailingStart, "failing_start")
    plan = hs.compile(graph)
    assert plan, plan.diagnostics
    before = time.monotonic()
    if case.startswith("hung"):
        if case == "hung_cleanup":
            release.set()
        with hs.run_async(plan) as session:
            assert entered.wait(5), "callback entry setup deadline expired"
            lifecycle._run_lifecycle(session, started, 0.01, 0.1)
        raise AssertionError("stuck callback unexpectedly returned")
    with hs.run_async(plan) as session:
        if not case.startswith("cpp") and case != "global_fail":
            assert entered.wait(5)
            assert not started.is_set()
            release.set()
        if case == "global_fail":
            release.set()
        if case.endswith("fail"):
            assert session.wait(timeout=5)
            assert session.termination is hs.RunTermination.FAILED
            assert any(d.token == "RUNTIME_LIFECYCLE_CALLBACK_FAILED" for d in session.diagnostics)
            assert not started.is_set()
        else:
            assert started.wait(5)
            if case == "cpp_delay":
                # Delay sanity only: setup/clock latency cannot prove the START barrier.
                assert time.monotonic() - before >= 0.25
            # A second clock activation neither completes the session nor changes ownership.
            assert not session.wait(timeout=1.1)
            if case == "isolation":
                other = threading.Event()
                other_graph = hs.Graph("fresh-observer")
                other_graph.set_default_clock(other_graph.add_clock(hs.RealtimeClock, "clock"))
                release.clear()
                entered.clear()
                other_graph.op(Predecessor, "predecessor")
                lifecycle.add_startup_observer(other_graph, other)
                with hs.run_async(hs.compile(other_graph)) as other_session:
                    assert entered.wait(5)
                    assert started.is_set()
                    assert not other.is_set()
                    release.set()
                    assert other.wait(5)
                    other_session.request_stop()
                    assert other_session.wait(timeout=5)
                assert not other_session.diagnostics
                assert other_session.termination is hs.RunTermination.STOPPED
            session.request_stop()
            assert session.wait(timeout=5)
            assert not session.diagnostics
            assert session.termination is hs.RunTermination.STOPPED


# REQ: PY-API-019
def test_deadline_overshoot_and_shutdown_timeout(capsys):
    timeline = Timeline(0.21)
    original_wait = timeline.wait
    timeline.wait = lambda timeout: original_wait(timeout + 0.2)
    session = Session(timeline)
    assert run(timeline, session) == 1
    assert not session.run_waits
    assert "startup timeout" in capsys.readouterr().err

    timeline = Timeline(0)
    session = Session(timeline)
    session.wait = lambda **_kwargs: False
    assert run(timeline, session) == 1
    assert "shutdown timeout" in capsys.readouterr().err


# REQ: PY-API-019
def test_event_observation_crossing_deadline_cannot_start_duration(capsys):
    timeline = Timeline(0.25)
    original_is_set = timeline.is_set

    def observe_after_deadline():
        timeline.now = 0.25
        return original_is_set()

    timeline.is_set = observe_after_deadline
    session = Session(timeline)
    assert run(timeline, session, startup_timeout=0.2) == 1
    assert not session.run_waits
    assert session.stopped
    assert "startup timeout" in capsys.readouterr().err


if __name__ == "__main__":
    hosted_case(sys.argv[1])
