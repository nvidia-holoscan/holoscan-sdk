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

STRUCTURED_FAILURE_SOURCE = "holoscan.tests.StructuredFailureSource"


def running_plan(name):
    """Compile the long-running graph used by every RunSession misuse test."""
    graph = hs.Graph(name)
    source = graph.op("holoscan.tests.CounterSource", "source", limit=1_000)
    sink = graph.op("holoscan.tests.Sink", "sink")
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
            "holoscan.tests.CounterSource", "source", limit=1_000_000
        )
        sink = graph.op("holoscan.tests.Sink", "sink")
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
