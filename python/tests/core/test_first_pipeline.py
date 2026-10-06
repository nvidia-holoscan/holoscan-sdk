# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Python compose, compile, and run acceptance test."""

import os
import signal
import subprocess
import threading

import holoscan_test_support

import holoscan as hs

COUNTER = "holoscan.tests.CounterSource"
DOUBLER = "holoscan.tests.DoubleIt"
SINK = "holoscan.tests.Sink"


def first_pipeline_graph(limit=10):
    """Compose the graph for the explicit-stop test.

    Representative: test_first_pipeline_runs_until_explicit_stop_without_signal_changes.
    """
    graph = hs.Graph("first-pipeline")
    source = graph.op(COUNTER, "source", limit=limit)
    doubler = graph.op(DOUBLER, "doubler")
    sink = graph.op(SINK, "sink")
    graph.add_flow(source.out, doubler["in"], queue_depth=4)
    graph.add_flow(doubler.out, sink["in"], queue_depth=4)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    return graph


def current_signal_state():
    """Snapshot signal state for the explicit-stop test.

    Representative: test_first_pipeline_runs_until_explicit_stop_without_signal_changes.
    """
    dispositions = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    return dispositions, mask


# REQ: PY-API-005
def test_first_pipeline_runs_until_explicit_stop_without_signal_changes():
    # Key setup: Compile the canonical pipeline and run it asynchronously under observation.
    #
    #   source --uint32--> doubler --uint32--> sink
    graph = first_pipeline_graph()
    plan = hs.compile(graph)
    assert plan

    helper = os.environ["HOLOSCAN_FIRST_PIPELINE_PLAN_HELPER"]
    cpp_json = subprocess.run(
        [helper],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    # Key assertion: Python and independently authored C++ produce the same immutable plan.
    assert plan.json() == cpp_json

    before_signals = current_signal_state()
    holoscan_test_support._reset_pipeline_observation()
    observed = threading.Event()

    def observe_ticks():
        if holoscan_test_support._wait_for_pipeline_observation(3, 5.0):
            observed.set()

    observer = threading.Thread(target=observe_ticks)
    observer.start()
    with hs.run_async(plan) as session:
        assert session.wait(timeout=0.0) is False
        assert observed.wait(timeout=5.0)
        assert holoscan_test_support._pipeline_observation_count() >= 3
        assert holoscan_test_support._pipeline_observation_last_value() >= 6
        session.request_stop()
        session.request_stop()
        assert session.wait()

    observer.join(timeout=5.0)

    # Key assertion: Explicit stop is terminal, clean, and leaves process signal state untouched.
    assert not observer.is_alive()
    assert session.termination is hs.RunTermination.STOPPED
    assert session.diagnostics == []
    assert current_signal_state() == before_signals
    assert "STOPPED" in repr(session)


# REQ: PY-API-005
def test_blocking_run_returns_terminal_session():
    # Key setup: Run a finite source/sink graph through the blocking API.
    #
    #   source --uint32--> sink
    graph = hs.Graph("blocking-run")
    source = graph.op("holoscan.tests.TickSource", "source")
    sink = graph.op("holoscan.tests.CompletingSink", "sink")
    graph.add_flow(source.out, sink["in"], queue_depth=4)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))

    session = hs.run(hs.compile(graph))

    # Key assertion: Blocking run returns the clean terminal session rather than an exit code.
    assert session.termination is hs.RunTermination.REQUESTED_COMPLETED
    assert session.diagnostics == []
