"""
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import contextlib
import gc
import tracemalloc

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

from .env_wrapper import env_var_context

# Total number of times the application is run during the test.
GLOBAL_RUN_COUNT = 2000
# Number of initial runs used only to reach a steady state before the baseline
# snapshot is taken. One-time allocations (imports, lazily-created singletons,
# internal caches) settle during warmup so they are not later mistaken for growth.
WARMUP_RUN_COUNT = 500
# Pass/fail is based on *growth* in the number of live allocations at a given
# source line between the post-warmup baseline snapshot and the final snapshot,
# not on an absolute count. A real leak in the repeated-run path accumulates
# roughly one live block per run, so over the (GLOBAL_RUN_COUNT - WARMUP_RUN_COUNT)
# measured runs a genuine leak would add ~1500 blocks, while a leak-free path
# stays essentially flat (growth near zero). This limit sits far above benign
# drift and far below a real linear leak.
#
# NOTE: this value was chosen by reasoning about the growth window, not measured
# on CI hardware (the SDK could not be run in the dev environment). Confirm/tune
# against real CI runs before relying on it as a tight bound.
GLOBAL_STAT_GROWTH_LIMIT = 500


class SinkOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        op_input.receive("in")


class PingTxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.index = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")
        spec.param("recess_period", 0)  # add parameter for test

    def compute(self, op_input, op_output, context):
        op_output.emit(self.index, "out")
        self.index += 1


class MyPingApp(Application):
    def compose(self):
        # Define the tx and rx operators, allowing tx to execute 10 times
        tx = PingTxOp(self, CountCondition(self, 10), name="tx")
        rx = SinkOp(self, name="rx")

        # Define the workflow:  tx -> rx
        self.add_flow(tx, rx)


def _assert_no_run_leak(run_once, label, capsys):
    """Run the app many times and assert no allocation site grows unboundedly.

    A memory/reference leak in the repeated-run path accumulates live
    allocations proportional to the number of runs. To distinguish a leak from a
    high-but-stable allocation site, this measures the *growth* in live-block
    count between a post-warmup baseline snapshot and a final snapshot, rather
    than an absolute count at a single point in time.

    Args:
        run_once: Callable that executes the application exactly once.
        label: Short description of the run mode, used in log output.
        capsys: pytest capture fixture; used to surface the growth report on the
            real stdout even when the test passes, without disabling capture for
            the rest of the run.
    """
    # Restrict tracing to Holoscan SDK code, dropping unrelated interpreter,
    # pytest, and tracemalloc churn that would otherwise add noise to the ranking.
    trace_filters = (tracemalloc.Filter(True, "*/holoscan/*"),)

    # log at warning level to make logs less verbose on failure
    env_var_settings = {
        ("HOLOSCAN_LOG_LEVEL", "WARN"),
    }
    tracemalloc.start()
    try:
        with env_var_context(env_var_settings):
            print(f"Running {WARMUP_RUN_COUNT} warmup cycles ({label})...")
            for _ in range(WARMUP_RUN_COUNT):
                run_once()
                gc.collect()
            baseline = tracemalloc.take_snapshot().filter_traces(trace_filters)

            measured_runs = GLOBAL_RUN_COUNT - WARMUP_RUN_COUNT
            print(f"Running {measured_runs} measured cycles ({label})...")
            for _ in range(measured_runs):
                run_once()
                gc.collect()
            final = tracemalloc.take_snapshot().filter_traces(trace_filters)
    finally:
        tracemalloc.stop()

    # Rank allocation sites by growth in live-block count between the snapshots.
    top_stats = final.compare_to(baseline, "lineno")
    top_stats.sort(key=lambda stat: stat.count_diff, reverse=True)

    # Surface the growth report on the real stdout even on success. Only this
    # block escapes pytest's capture; app log output stays suppressed via
    # HOLOSCAN_LOG_LEVEL=WARN and other tests' output is unaffected.
    is_passed = True
    with capsys.disabled():
        print(f"[ Top 10 growth sites over {measured_runs} runs ]")
        for stat in top_stats[:10]:
            print(stat)
            if stat.count_diff > GLOBAL_STAT_GROWTH_LIMIT:
                print(
                    f"warning: stat.count_diff: {stat.count_diff} exceeds the limit: "
                    f"{GLOBAL_STAT_GROWTH_LIMIT} over {measured_runs} runs"
                )
                print(f"warning: stat.traceback: {stat.traceback}")
                is_passed = False
    assert is_passed, (
        f"Potential memory leak in the {label} run path: an allocation site grew by more than "
        f"{GLOBAL_STAT_GROWTH_LIMIT} live blocks over {measured_runs} repeated runs "
        f"(see 'Top 10 growth sites' above)."
    )


def test_multiple_run(capsys):
    app = MyPingApp()
    _assert_no_run_leak(app.run, label="sync", capsys=capsys)


def test_multiple_run_async(capsys):
    app = MyPingApp()

    def run_once():
        app.run_async().result()

    _assert_no_run_leak(run_once, label="async", capsys=capsys)


class _NullCapsys:
    """Minimal stand-in for the pytest capsys fixture for direct execution."""

    @contextlib.contextmanager
    def disabled(self):
        yield


def main():
    test_multiple_run(_NullCapsys())
    test_multiple_run_async(_NullCapsys())


if __name__ == "__main__":
    main()
