# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-installed strict lifecycle qualification for the shipped CuPy graph."""

import argparse
import importlib.util
import math
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples/operators/holoviz_cupy_pattern/python/holoviz_cupy_pattern.py"
)
_SPEC = importlib.util.spec_from_file_location("holoviz_cupy_pattern", _SOURCE)
example = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = example
_SPEC.loader.exec_module(example)
hs = example.hs


def observe_emissions(source_type, started):
    """Observe successful source publications without changing the shipped tutorial."""

    class ObservedSource(source_type):
        def compute(self, context):
            before = self.frame
            super().compute(context)
            if self.frame != before:
                started.set()

    return ObservedSource


def run_lifecycle(session, started, seconds, startup_timeout, *, monotonic=time.monotonic):
    """Wait for an emission, measure the run, then stop and inspect the authoritative result.

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
            # The event proves an emission, not that rendering or the run succeeded.
            if session.wait(timeout=0):
                failure = "session ended during startup confirmation"
            break
        remaining = deadline - now
        if remaining <= 0:
            failure = (
                f"startup timeout after {startup_timeout:g}s; first Image emission unconfirmed"
            )
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

    result = example.final_result(session)
    return result if result else int(failure is not None)


def main(argv=None):
    """Run the real graph with a strict first-emission deadline and measured duration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    startup, remainder = parser.parse_known_args(argv)
    if not math.isfinite(startup.startup_timeout) or startup.startup_timeout <= 0:
        parser.error("--startup-timeout must be finite and positive")
    config = example.parse_args(remainder)
    try:
        cupy = example.load_cupy()
        example.require_single_device(cupy)
    except example.UnavailableError as error:
        print(f"skipping: {error}")
        return example.SKIP_EXIT_CODE
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    started = threading.Event()
    source_factory = example.pattern_source
    with patch.object(
        example,
        "pattern_source",
        lambda config, cupy: observe_emissions(source_factory(config, cupy), started),
    ):
        graph = example.build_graph(config, cupy)
    plan = hs.compile(graph, example.compile_options())
    if not plan:
        print(plan.summary(), file=sys.stderr)
        return 1
    with hs.run_async(plan) as session:
        result = run_lifecycle(session, started, config.seconds, startup.startup_timeout)
    if result == 0:
        print(f"emitted Images, ran {config.seconds:g}s after startup, and stopped cleanly")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
