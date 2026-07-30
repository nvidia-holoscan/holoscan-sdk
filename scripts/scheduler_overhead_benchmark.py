#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Measures scheduling and message-passing overhead for Holoscan operators.

This script benchmarks the performance of different scheduler configurations
to help developers understand operator granularity trade-offs.

Usage:
    HOLOSCAN_LOG_LEVEL=ERROR python3 scheduler_overhead_benchmark.py

The script compares:
- Greedy vs. Event-Based schedulers
- Operator execution with and without message passing
- Different worker thread configurations

See the Performance Considerations documentation for interpretation guidance:
https://docs.nvidia.com/holoscan/sdk-user-guide/performance_considerations.html
"""

import argparse
import time

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler


class MyOperator(Operator):
    """A minimal operator with no input/output ports (no messaging overhead)."""

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.count = 0

    def compute(self, op_input, op_output, context):
        self.count += 1


class PingTxOp(Operator):
    """A transmitter operator that emits a message each iteration."""

    def __init__(self, fragment, *args, **kwargs):
        self.count = 0
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        self.count += 1
        op_output.emit(self.count, "out")


class PingRxOp(Operator):
    """A receiver operator that receives a message each iteration."""

    def __init__(self, fragment, *args, **kwargs):
        self.count = 0
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        _ = op_input.receive("in")
        self.count += 1


class MyOperatorRunnerApp(Application):
    """Application with two independent operators (no message passing)."""

    def __init__(self, iterations=10):
        self.iterations = iterations
        super().__init__()

        self.op1 = None
        self.op2 = None

    def compose(self):
        self.op1 = MyOperator(self, CountCondition(self, self.iterations), name="op1")
        self.op2 = MyOperator(self, CountCondition(self, self.iterations), name="op2")
        self.add_operator(self.op1)
        self.add_operator(self.op2)

    @property
    def count(self):
        return self.op1.count + self.op2.count


class MyPingApp(Application):
    """Application with message passing between Tx and Rx operators."""

    def __init__(self, iterations=10):
        self.iterations = iterations
        self.tx = None
        self.rx = None
        super().__init__()

    def compose(self):
        # Define the tx and rx operators, allowing tx to execute `iterations` times
        self.tx = PingTxOp(self, CountCondition(self, self.iterations), name="tx")
        self.rx = PingRxOp(self, name="rx")

        # Define the workflow: tx -> rx
        self.add_flow(self.tx, self.rx)

    @property
    def count(self):
        return self.tx.count + self.rx.count


def run_app(
    app_class: type[Application], scheduler: str, iterations: int, num_workers: int = 1
) -> int:
    """Run an application with the specified configuration."""
    app = app_class(iterations=iterations)

    if scheduler == "greedy":
        app.scheduler(GreedyScheduler(app))
    elif scheduler == "event_based":
        # Setting 'stop_on_deadlock_timeout=0' (the default value) may lead to
        # incorrect deadlock detection.
        app.scheduler(
            EventBasedScheduler(app, worker_thread_number=num_workers, stop_on_deadlock_timeout=1)
        )
    else:
        raise ValueError(f"Invalid scheduler: {scheduler}")

    app.run()
    return app.count


def benchmark(
    app_class: type[Application], scheduler: str, iterations: int, num_workers: int = 1
) -> tuple[float, int, float]:
    """Run a benchmark and return (elapsed_seconds, count, us_per_iteration)."""
    start_time = time.perf_counter()
    count = run_app(app_class, scheduler, iterations=iterations, num_workers=num_workers)
    end_time = time.perf_counter()

    elapsed_s = end_time - start_time
    us_per_iter = elapsed_s / iterations * 1_000_000.0
    return elapsed_s, count, us_per_iter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microbenchmark for Holoscan scheduler + messaging overheads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100000,
        help="Number of iterations per benchmark run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Event-based scheduler worker thread counts to test.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=100,
        help="Warm-up iterations run before benchmarking (set to 0 to disable).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if any(w <= 0 for w in args.workers):
        raise ValueError("--workers values must be > 0")

    # Warming up (optional)
    if args.warmup_iterations > 0:
        run_app(MyOperatorRunnerApp, "greedy", iterations=args.warmup_iterations)
        run_app(
            MyOperatorRunnerApp, "event_based", iterations=args.warmup_iterations, num_workers=2
        )
        run_app(MyPingApp, "greedy", iterations=args.warmup_iterations)
        run_app(MyPingApp, "event_based", iterations=args.warmup_iterations, num_workers=2)

    for num_workers in args.workers:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"Number of workers: {num_workers}")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        for app_class in [MyOperatorRunnerApp, MyPingApp]:
            print("################################")
            print(f"App class: {app_class.__name__}")
            print("################################")

            for scheduler in ["greedy", "event_based"]:
                elapsed_s, count, us_per_iter = benchmark(
                    app_class, scheduler, iterations=args.iterations, num_workers=num_workers
                )

                print(f"Scheduler: {scheduler}")
                print(f"Iterations: {args.iterations}")
                if scheduler == "event_based":
                    print(f"Worker thread number: {num_workers}")
                else:
                    print("Worker thread number: 1")
                print(f"Execution time: {elapsed_s:.4f} seconds")
                print(f"Count: {count}")
                print(f"Execution time per iteration (us): {us_per_iter:.4f} us")
                print("--------------------------------")


if __name__ == "__main__":
    main()
