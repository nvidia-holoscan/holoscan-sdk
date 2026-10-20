# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RunSession lifetime and misuse checks."""

import gc
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import holoscan_test_support
import pytest

import holoscan as hs

STRUCTURED_FAILURE_SOURCE = "holoscan.test.StructuredFailureSource"


def running_plan(name):
    """Compile the long-running graph used by every RunSession misuse test."""
    graph = hs.Graph(name)
    source = graph.op("holoscan.test.CounterSource", "source", limit=1_000)
    sink = graph.op("holoscan.test.Sink", "sink")
    graph.add_flow(source.out, sink["in"], queue_depth=4)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    return hs.compile(graph)


# REQ: PY-API-005
def test_session_without_with_cleans_up_and_double_stop_is_safe():
    # Key setup: Exercise destructor cleanup, bounded wait, and repeated stop on running sessions.
    #
    #   source --uint32--> sink
    holoscan_test_support._reset_pipeline_observation()
    bare_session = hs.run_async(running_plan("bare-session"))
    assert holoscan_test_support._wait_for_pipeline_observation(1, 5.0)
    del bare_session
    gc.collect()

    holoscan_test_support._reset_pipeline_observation()
    session = hs.run_async(running_plan("double-stop"))
    assert holoscan_test_support._wait_for_pipeline_observation(1, 5.0)
    progressed = threading.Event()

    def make_progress():
        time.sleep(0.01)
        progressed.set()

    worker = threading.Thread(target=make_progress)
    worker.start()
    assert session.wait(timeout=0.1) is False
    assert progressed.is_set()
    worker.join()
    session.request_stop()
    session.request_stop()

    # Key assertion: Repeated stop converges on one clean stopped terminal state.
    assert session.wait(timeout=5.0)
    assert session.termination is hs.RunTermination.STOPPED


# REQ: PY-API-005
def test_wait_rejects_invalid_timeout_values():
    # Key setup: Call bounded wait with negative and non-finite timeouts on a running session.
    #
    #   source --uint32--> sink
    session = hs.run_async(running_plan("invalid-timeout"))
    try:
        # Key assertion: Both invalid timeout shapes are rejected before waiting.
        with pytest.raises(ValueError, match="finite nonnegative"):
            session.wait(timeout=-1.0)
        with pytest.raises(ValueError, match="finite nonnegative"):
            session.wait(timeout=float("nan"))
    finally:
        session.request_stop()
        session.wait()


# REQ: PY-API-005
def test_context_manager_cleans_up_without_suppressing_exceptions():
    # Key setup: Raise an application exception inside an asynchronous session context.
    #
    #   source --uint32--> sink
    with (
        pytest.raises(RuntimeError, match="application failure"),
        hs.run_async(running_plan("context-exception")) as session,
    ):
        raise RuntimeError("application failure")

    # Key assertion: Cleanup stops the session without suppressing the application exception.
    assert session.termination is hs.RunTermination.STOPPED


# REQ: PY-API-005
def test_failed_session_exposes_structured_diagnostics_and_bounded_summary():
    # Key setup: Run a source whose compute callback raises one nested Core failure.
    #
    #   source
    graph = hs.Graph("structured-runtime-failure")
    graph.op(STRUCTURED_FAILURE_SOURCE, "source")
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    plan = hs.compile(graph)
    session = hs.run(plan)

    # Key assertion: Runtime evidence remains typed while the report uses Core's bounded formatter.
    assert session.termination is hs.RunTermination.FAILED
    diagnostic = next(
        row for row in session.diagnostics if row.token == "RUNTIME_OPERATOR_CALLBACK_FAILED"
    )
    assert diagnostic.detail.operation is hs.diagnostics.DiagnosticOperation.COMPUTE
    assert diagnostic.detail.role is hs.diagnostics.DiagnosticFailureRole.PRIMARY
    assert diagnostic.detail.operator_path == "source"
    assert diagnostic.detail.error.code is hs.errors.ErrorCode.INVALID_ARGUMENT
    assert diagnostic.detail.error.message == "primary runtime failure"
    assert len(diagnostic.detail.causes) == 1
    assert diagnostic.detail.causes[0].code is hs.errors.ErrorCode.NOT_FOUND
    assert diagnostic.detail.causes[0].message == "nested runtime cause"
    assert "diagnostic_index=0" in session.summary()
    assert "RUNTIME_OPERATOR_CALLBACK_FAILED" in session.summary()


# REQ: PY-API-005
def test_blocking_run_processes_python_interrupts():
    # Key setup: Interrupt a subprocess while its main thread waits in hs.run().
    script = textwrap.dedent(
        """
        import holoscan_test_support
        import holoscan as hs

        graph = hs.Graph("interruptible-run")
        source = graph.op(
            "holoscan.test.CounterSource", "source", limit=1_000_000
        )
        sink = graph.op("holoscan.test.Sink", "sink")
        graph.add_flow(source.out, sink["in"], queue_depth=4)
        graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
        plan = hs.compile(graph)
        try:
            print("READY", flush=True)
            hs.run(plan)
        except KeyboardInterrupt:
            print("INTERRUPTED", flush=True)
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "READY"
    os.kill(process.pid, signal.SIGINT)
    stdout, stderr = process.communicate(timeout=10)

    # Key assertion: Polling reacquires the GIL and turns SIGINT into KeyboardInterrupt.
    assert process.returncode == 0, stderr
    assert "INTERRUPTED" in stdout


# REQ: PY-API-035
def test_interpreter_exit_stops_sessions_python_still_owns():
    # Key setup: Exit the interpreter while a still-referenced session runs a hosted callback every
    # millisecond. The script's own atexit hook is registered before holoscan is imported, so it
    # runs after holoscan's exit hooks: it reports the session's state, then runs one more graph
    # whose hosted callbacks start only after the exit drain.
    script = textwrap.dedent(
        """
        import atexit
        import time

        def report():
            print(f"stopped={not SESSION.is_running}", flush=True)
            late = hs.run(hs.compile(late_graph))
            print(f"late_failed={late.termination is hs.RunTermination.FAILED}", flush=True)
            print(f"late_token={'PYTHON_INTERPRETER_UNAVAILABLE' in late.summary()}", flush=True)

        atexit.register(report)

        import holoscan as hs

        class Tick(hs.Operator):
            calls = 0

            def contract(self):
                return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

            def compute(self, context):
                type(self).calls += 1

        graph = hs.Graph("running-at-exit", default_clock=hs.RealtimeClock)
        graph.op(Tick, "tick")
        late_graph = hs.Graph("started-after-exit-drain", default_clock=hs.RealtimeClock)
        late_graph.op(Tick, "tick")
        SESSION = hs.run_async(hs.compile(graph))
        while Tick.calls < 20:
            time.sleep(0.001)
        print("running", flush=True)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    # Key assertion: The exit drain stops and joins the running session before finalization, and a
    # session started later fails its callbacks with an SDK token instead of entering Python.
    assert completed.returncode == 0, completed.stderr
    assert "terminate called" not in completed.stderr
    assert "running" in completed.stdout
    assert "stopped=True" in completed.stdout
    assert "late_failed=True" in completed.stdout
    assert "late_token=True" in completed.stdout


# REQ: PY-API-035
@pytest.mark.parametrize("mode", ["blocking", "asynchronous"])
def test_interpreter_exit_rejects_late_operator_reconstruction(mode):
    # Key setup: Compile a hosted graph on the main thread after the SDK exit hooks, then run it
    # synchronously or return from the hook with its asynchronous session still referenced.
    script = textwrap.dedent(
        """
        import atexit
        import threading

        main_thread = threading.get_native_id()
        phase = "initial"
        constructors = []

        def late_run():
            global phase, LATE_SESSION
            phase = "late"
            graph = hs.Graph("late-reconstruction", default_clock=hs.RealtimeClock)
            graph.op(Tick, "tick")
            plan = hs.compile(graph)
            assert plan.ok
            assert any(tid == main_thread for tid in constructors)
            before = len(constructors)
            if MODE == "blocking":
                session = hs.run(plan)
                assert session.termination is hs.RunTermination.FAILED
                assert "PYTHON_INTERPRETER_UNAVAILABLE" in session.summary()
                assert constructors[before:] == []
                print("late_reconstruction_rejected", flush=True)
            else:
                LATE_SESSION = hs.run_async(plan)
                print("late_async_returned", flush=True)

        atexit.register(late_run)
        import holoscan as hs

        class Tick(hs.Operator):
            def __init__(self):
                thread = threading.get_native_id()
                constructors.append(thread)
                if phase == "late" and thread != main_thread:
                    print("LATE_WORKER_CONSTRUCTOR", flush=True)

            def contract(self):
                return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

            def compute(self, context):
                context.request_stop()

        """
    ).replace("MODE", repr(mode))
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    # Key assertion: Main-thread reconstruction works, but a late Runtime constructor never runs.
    # The asynchronous case returns to interpreter finalization without waiting or choosing a
    # worker interleaving with a sleep; the blocking case observes the worker's failed entry.
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "LATE_WORKER_CONSTRUCTOR" not in completed.stdout
    expected = "late_reconstruction_rejected" if mode == "blocking" else "late_async_returned"
    assert expected in completed.stdout, completed.stdout + completed.stderr


def drain_seconds(stdout):
    """Return the exit drain duration printed by the subprocess scripts below."""
    lines = [line for line in stdout.splitlines() if line.startswith("drain=")]
    assert len(lines) == 1, stdout
    return float(lines[0].removeprefix("drain="))


# REQ: PY-API-035
def test_interpreter_exit_drain_shares_one_bound_across_owned_and_abandoned_sessions():
    # Key setup: At exit, a still-referenced session and a session already abandoned to the reaper
    # both run a callback that outlasts the exit bound. Hooks registered around holoscan's measure
    # its drain; the last one exits at once, so the sleeping callbacks never meet finalization.
    script = textwrap.dedent(
        """
        import atexit
        import os
        import threading
        import time

        marks = {}

        def finish():
            print(f"drain={time.monotonic() - marks['start']:.2f}", flush=True)
            os._exit(0)

        atexit.register(finish)

        import holoscan as hs

        abandoned_running = threading.Event()
        abandoned = threading.Event()
        HOLDER = []

        class Outlast(hs.Operator):
            def contract(self):
                return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

            def compute(self, context):
                abandoned_running.set()
                time.sleep(30)

        class Abandon(hs.Operator):
            def contract(self):
                return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

            def compute(self, context):
                abandoned_running.wait(10)
                HOLDER.clear()  # destroyed in a callback, so the reaper takes the session
                abandoned.set()
                time.sleep(30)

        def plan(name, operator):
            graph = hs.Graph(name, default_clock=hs.RealtimeClock)
            graph.op(operator, "op")
            return hs.compile(graph)

        HOLDER.append(hs.run_async(plan("abandoned-at-exit", Outlast)))
        SESSION = hs.run_async(plan("owned-at-exit", Abandon))
        assert abandoned.wait(10)
        atexit.register(lambda: marks.__setitem__("start", time.monotonic()))
        print("ready", flush=True)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    # Key assertion: Both populations wait within one 5 s exit bound, not one bound each.
    assert completed.returncode == 0, completed.stderr
    assert "ready" in completed.stdout
    assert 4.5 <= drain_seconds(completed.stdout) <= 7.5, completed.stdout


# REQ: PY-API-035
def test_interpreter_exit_drain_lets_a_callback_release_its_own_session():
    # Key setup: A hosted callback blocks until the exit drain starts, gives the drain time to begin
    # waiting for its session, then drops the last reference to that session. A second session
    # ticks every millisecond, so its callbacks need the GIL while the first one is released.
    script = textwrap.dedent(
        """
        import atexit
        import threading
        import time

        marks = {}

        def finish():
            print(f"drain={time.monotonic() - marks['start']:.2f}", flush=True)
            print(f"released={released.is_set()}", flush=True)
            print(f"other_stopped={not OTHER.is_running}", flush=True)

        atexit.register(finish)

        import holoscan as hs

        drain_started = threading.Event()
        released = threading.Event()
        HOLDER = []

        class ReleaseOwnSession(hs.Operator):
            def contract(self):
                return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

            def compute(self, context):
                if released.is_set():
                    return
                drain_started.wait(20)
                time.sleep(0.5)  # the drain is now waiting for this session
                HOLDER.clear()
                released.set()

        class Tick(hs.Operator):
            calls = 0

            def contract(self):
                return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

            def compute(self, context):
                type(self).calls += 1

        def plan(name, operator):
            graph = hs.Graph(name, default_clock=hs.RealtimeClock)
            graph.op(operator, "op")
            return hs.compile(graph)

        HOLDER.append(hs.run_async(plan("released-in-callback", ReleaseOwnSession)))
        OTHER = hs.run_async(plan("ticking", Tick))
        while Tick.calls < 20:
            time.sleep(0.001)

        def start():
            marks["start"] = time.monotonic()
            drain_started.set()

        atexit.register(start)
        print("ready", flush=True)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    # Key assertion: The callback releases its session without waiting for the drain, and both
    # sessions stop and join well within the exit bound.
    assert completed.returncode == 0, completed.stderr
    assert "terminate called" not in completed.stderr
    assert "ready" in completed.stdout
    assert "released=True" in completed.stdout
    assert "other_stopped=True" in completed.stdout
    assert drain_seconds(completed.stdout) < 3.0, completed.stdout


# REQ: PY-API-035
@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_while_the_live_session_registry_is_held_leaves_the_child_usable():
    # Key setup: Fork while another thread holds the registry of sessions Python owns. The child
    # then destroys the session it inherited, runs a session of its own, and exits normally, which
    # runs holoscan's exit drain. A child that inherited a locked registry would hang until its
    # alarm fires.
    script = textwrap.dedent(
        """
        import os
        import signal
        import sys
        import threading
        import warnings

        import holoscan as hs
        from holoscan._core import _test_hold_live_session_registry

        warnings.simplefilter("ignore", DeprecationWarning)  # fork() with threads running

        class Tick(hs.Operator):
            def contract(self):
                return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

            def compute(self, context):
                pass

        class Brief(hs.Operator):
            calls = 0

            def contract(self):
                return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

            def compute(self, context):
                type(self).calls += 1
                if type(self).calls == 3:
                    context.request_stop()

        def plan(name, operator):
            graph = hs.Graph(name, default_clock=hs.RealtimeClock)
            graph.op(operator, "op")
            return hs.compile(graph)

        PARENT = hs.run_async(plan("parent", Tick))
        child_plan = plan("child", Brief)
        held = threading.Event()
        holder = threading.Thread(target=_test_hold_live_session_registry, args=(0.5, held))
        holder.start()
        assert held.wait(5)
        child = os.fork()
        if child == 0:
            signal.alarm(20)
            del PARENT  # inherited: released without touching the registry
            session = hs.run(child_plan)
            print(f"child_ran={session.termination is hs.RunTermination.STOPPED}", flush=True)
            del session
            sys.exit(0)
        holder.join()
        _, status = os.waitpid(child, 0)
        print(f"child_exit={os.waitstatus_to_exitcode(status)}", flush=True)
        print(f"parent_running={PARENT.is_running}", flush=True)
        PARENT.request_stop()
        print(f"parent_stopped={PARENT.wait(timeout=5)}", flush=True)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    # Key assertion: The child creates, destroys, and drains sessions normally, and the parent's
    # session is unaffected.
    assert completed.returncode == 0, completed.stderr
    assert "child_ran=True" in completed.stdout, completed.stderr
    assert "child_exit=0" in completed.stdout, completed.stdout
    assert "parent_running=True" in completed.stdout
    assert "parent_stopped=True" in completed.stdout
