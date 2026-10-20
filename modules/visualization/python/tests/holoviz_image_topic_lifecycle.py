# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-installed global START qualification for the shipped image-topic graph.

The example remains a session-duration smoke. This driver adds a run-local,
clock-triggered observer whose first compute follows the global START barrier.
An outer process watchdog must bound native callbacks and context-manager cleanup.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import math
import sys
import threading
import time
from pathlib import Path

import holoscan as hs

_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/operators/holoviz_image_topic/python/holoviz_image_topic.py"
)
_SPEC = importlib.util.spec_from_file_location("holoviz_image_topic", _EXAMPLE_PATH)
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def add_startup_observer(graph, started):
    """Add a fresh hosted observer using the graph's existing default clock."""

    class StartupObserver(hs.Operator):
        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=datetime.timedelta(seconds=1)))

        def compute(self, context):
            started.set()

    return graph.op(StartupObserver, "test_startup_observer")


def _run_lifecycle(session, started, seconds, startup_timeout, *, monotonic=time.monotonic):
    """Wait for START, measure the run, then stop and inspect the authoritative result.

    Polling and the stop wait are bounded; native START and context-manager cleanup may
    still block. Callers/tests need an outer watchdog to bound the entire process.
    """
    deadline = monotonic() + startup_timeout
    failure = None
    while True:
        if session.wait(timeout=0):
            failure = "session ended before startup confirmation"
            break
        observed_start = started.is_set()
        now = monotonic()
        if observed_start and now <= deadline:
            # Compute follows the global START barrier; final failure still takes precedence.
            if session.wait(timeout=0):
                failure = "session ended during startup confirmation"
            break
        remaining = deadline - now
        if remaining <= 0:
            failure = f"startup timeout after {startup_timeout:g}s; global START unconfirmed"
            break
        started.wait(min(0.05, remaining))

    if failure is None and session.wait(timeout=seconds):
        failure = f"session ended before the requested {seconds:g}s"
    if failure is not None:
        # Flush before cooperative stop / join, which cannot interrupt a native callback.
        print(failure, file=sys.stderr, flush=True)
    session.request_stop()
    if not session.wait(timeout=10.0):
        print("shutdown timeout after 10s; cleanup may still block", file=sys.stderr, flush=True)
        return 1

    if session.diagnostics:
        if example.display_unavailable(session):
            print("no display available; skipping")
            return example.SKIP_EXIT_CODE
        for diagnostic in session.diagnostics:
            print(f"run failed: {diagnostic.token}", file=sys.stderr, flush=True)
        return 1
    if session.termination is not hs.RunTermination.STOPPED:
        print(f"unexpected run termination: {session.termination}", file=sys.stderr, flush=True)
        return 1
    return 1 if failure is not None else 0


# REQ: PY-API-019
# REQ: LIFECYCLE-001
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=30.0,
        help="global START observation budget in seconds (default: 30)",
    )
    arguments = parser.parse_args()
    for option, value in (
        ("--seconds", arguments.seconds),
        ("--startup-timeout", arguments.startup_timeout),
    ):
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"invalid {option} value: '{value:g}'")

    graph = example.build_graph()
    started = threading.Event()
    add_startup_observer(graph, started)
    plan = hs.compile(graph)
    if not plan:
        for diagnostic in plan.diagnostics:
            print(f"compile rejected: {diagnostic.token}", file=sys.stderr, flush=True)
        return 1
    with hs.run_async(plan) as session:
        result = _run_lifecycle(session, started, arguments.seconds, arguments.startup_timeout)
    if result == 0:
        print(f"global START observed, ran for {arguments.seconds:g}s and stopped cleanly")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
