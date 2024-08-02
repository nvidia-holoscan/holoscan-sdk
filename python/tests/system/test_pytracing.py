"""
 SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""  # noqa: E501

import sys

from holoscan.conditions import CountCondition
from holoscan.core import Application, IOSpec, Operator, OperatorSpec
from holoscan.schedulers import EventBasedScheduler, MultiThreadScheduler

# The following example is extracted from the ping_vector example in the public/examples
# directory (public/examples/ping_vector/python/ping_vector.py), adding some methods
# (e.g., initialize, start, stop) to the operators to demonstrate that the methods are
# called.


class PingTxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.index = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def initialize(self):
        print("Tx initialize")

    def start(self):
        print("Tx start")

    def stop(self):
        print("Tx stop")

    def compute(self, op_input, op_output, context):
        value = self.index
        self.index += 1

        output = []
        for _ in range(0, 5):
            output.append(value)
            value += 1

        op_output.emit(output, "out")


class PingMxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.count = 1

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out1")
        spec.output("out2")
        spec.output("out3")
        spec.param("multiplier", 2)

    def initialize(self):
        print("Mx initialize")

    def start(self):
        print("Mx start")

    def stop(self):
        print("Mx stop")

    def compute(self, op_input, op_output, context):
        values1 = op_input.receive("in")
        print(f"Middle message received (count: {self.count})")
        self.count += 1

        values2 = []
        values3 = []
        for i in range(0, len(values1)):
            print(f"Middle message value: {values1[i]}")
            values2.append(values1[i] * self.multiplier)
            values3.append(values1[i] * self.multiplier * self.multiplier)

        op_output.emit(values1, "out1")
        op_output.emit(values2, "out2")
        op_output.emit(values3, "out3")


class PingRxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.count = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.input("dup_in")
        spec.input("receivers", size=IOSpec.ANY_SIZE)

    def initialize(self):
        print("Rx initialize")

    def start(self):
        print("Rx start")

    def stop(self):
        print("Rx stop")

    def compute(self, op_input, op_output, context):
        receiver_vector = op_input.receive("receivers")
        input_vector = op_input.receive("in")
        dup_input_vector = op_input.receive("dup_in")
        print(
            f"Rx message received (count: {self.count}, input vector size: {len(input_vector)},"
            "duplicated input vector size: {len(dup_input_vector)}, receiver size: "
            "{len(receiver_vector)})"
        )
        self.count += 1

        for i in range(0, len(input_vector)):
            print(f"Rx message input value[{i}]: {input_vector[i]}")

        for i in range(0, len(dup_input_vector)):
            print(f"Rx message duplicated input value[{i}]: {dup_input_vector[i]}")

        for i in range(0, len(receiver_vector)):
            for j in range(0, len(receiver_vector[i])):
                print(f"Rx message receiver value[{i}][{j}]: {receiver_vector[i][j]}")


class MyPingApp(Application):
    def compose(self):
        # Define the tx, mx, rx operators, allowing the tx operator to execute 3 times
        tx = PingTxOp(self, CountCondition(self, 3), name="tx")
        mx = PingMxOp(self, name="mx", multiplier=3)
        rx = PingRxOp(self, name="rx")

        # Define the workflow
        self.add_flow(tx, mx, {("out", "in")})
        self.add_flow(mx, rx, {("out1", "in"), ("out1", "dup_in")})
        self.add_flow(mx, rx, {("out2", "receivers"), ("out3", "receivers")})


def main(scheduler_type="greedy"):
    app = MyPingApp()
    if scheduler_type in ["multithread", "event_based"]:
        # If a multithread scheduler is used, the cProfile or profile module may not work
        # properly and show some error messages.
        # For multithread scheduler, please use multithread-aware profilers such as
        # [pyinstrument](https://github.com/joerick/pyinstrument),
        # [pprofile](https://github.com/vpelletier/pprofile), or
        # [yappi](https://github.com/sumerc/yappi).
        if scheduler_type == "multithread":
            scheduler_class = MultiThreadScheduler
            name = ("multithread_scheduler",)
        else:
            scheduler_class = EventBasedScheduler
            name = ("event_based_scheduler",)
        scheduler = scheduler_class(
            app,
            worker_thread_number=3,
            stop_on_deadlock=True,
            stop_on_deadlock_timeout=500,
            name=name,
        )
        app.scheduler(scheduler)

    app.run()


def verify_ncalls(pstats, func_name, expected_ncalls, func_count=1):
    count = 0
    for key in pstats.stats:
        if key[0] == __file__ and key[2] == func_name:
            count += 1
            print(key, pstats.stats[key])
            # See https://github.com/python/cpython/blob/6f23472345aedbba414620561ba89fa3cf6eda24/
            # Lib/pstats.py#L202 to understand the structure of the tuple
            # (pcalls, ncalls, ...)
            assert pstats.stats[key][1] == expected_ncalls
    assert count == func_count


def profile_main(scheduler_type="greedy"):
    import pstats
    from cProfile import Profile

    pr = Profile()
    pr.runcall(main, scheduler_type)

    stats = pstats.Stats(pr)

    verify_ncalls(stats, "initialize", 1, 3)
    verify_ncalls(stats, "start", 1, 3)
    verify_ncalls(stats, "stop", 1, 3)
    verify_ncalls(stats, "compute", 3, 3)


def verify_traced_funcs(traced_funcs, should_exists):
    for func in should_exists:
        print(f"Checking if '{func}' exists")
        assert func in traced_funcs


def trace_main(scheduler_type="greedy"):
    import trace

    tracer = trace.Trace(
        ignoredirs=[sys.prefix, sys.exec_prefix],
        trace=0,
        count=0,
        countfuncs=0,
        countcallers=1,
        timing=True,
    )
    tracer.runfunc(main, scheduler_type)
    r = tracer.results()
    traced_funcs = set()
    callers = r.callers
    for (pfile, pmod, pfunc), (_cfile, cmod, cfunc) in sorted(callers):
        if pfile != __file__:
            continue
        traced_funcs.add(cfunc)
        print(f"  {pmod}.{pfunc} -> {cmod}.{cfunc}")

    should_exists = (
        "MyPingApp.compose",
        "PingMxOp.compute",
        "PingMxOp.initialize",
        "PingMxOp.start",
        "PingMxOp.stop",
        "PingRxOp.compute",
        "PingRxOp.initialize",
        "PingRxOp.start",
        "PingRxOp.stop",
        "PingTxOp.compute",
        "PingTxOp.initialize",
        "PingTxOp.start",
        "PingTxOp.stop",
    )
    verify_traced_funcs(traced_funcs, should_exists)


def verify_covered_funcs(covered_lines, should_exists):
    import inspect

    for func in should_exists:
        func_code, func_start_line = inspect.getsourcelines(func)
        print(
            f"Checking if one of the first two lines ({func_start_line+1}-{func_start_line+2}) of"
            f" '{func.__qualname__}' is covered."
        )
        # Check if the first two lines of the function are covered
        assert (func_start_line + 1 in covered_lines) or (func_start_line + 2 in covered_lines)


def coverage_main(scheduler_type="greedy"):
    try:
        from coverage import Coverage
    except ImportError:
        print("coverage module is not installed. Skipping the execution of this method.")
        return

    cov = Coverage()
    cov.start()

    main(scheduler_type)

    cov.stop()

    # Get the coverage report for the current file
    r = cov.get_data()
    covered_lines = sorted(r.lines(__file__))
    print(f"Covered lines: {covered_lines}")
    covered_lines = set(covered_lines)

    should_exists = (
        PingMxOp.compute,
        PingMxOp.initialize,
        PingMxOp.start,
        PingMxOp.stop,
        PingRxOp.compute,
        PingRxOp.initialize,
        PingRxOp.start,
        PingRxOp.stop,
        PingTxOp.compute,
        PingTxOp.initialize,
        PingTxOp.start,
        PingTxOp.stop,
    )
    verify_covered_funcs(covered_lines, should_exists)


def pdb_main(scheduler_type="greedy"):
    print(
        """
This is an interactive session.
Please type the following commands to check if the breakpoints are hit.

  (Pdb) b test_pytracing.py:76
  Breakpoint 1 at /workspace/holoscan-sdk/public/python/tests/system/test_pytracing.py:76
  (Pdb) c
  ...
  > /workspace/holoscan-sdk/public/python/tests/system/test_pytracing.py(76)start()
  -> print("Mx start")
  (Pdb) exit
"""
    )

    import pdb  # noqa: T100

    pdb.set_trace()  # noqa: T100

    main(scheduler_type)


def yappi_main(scheduler_type="greedy"):
    try:
        import yappi
    except ImportError:
        print("yappi module is not installed. Skipping the execution of this method.")
        return

    # Set the context ID callback to obtain the thread ID.
    # Without this, multiple contexts will be created for the same thread in the Holoscan
    # application.
    import _thread

    yappi.set_context_id_callback(lambda: _thread.get_ident())

    yappi.start()
    main(scheduler_type)
    yappi.stop()

    threads = yappi.get_thread_stats()
    threads.print_all()

    for thread in threads:
        print("# Function stats for (%s) (%d)" % (thread.name, thread.id))

        stats = yappi.get_func_stats(ctx_id=thread.id)
        for stat in stats:
            print(
                "  %s %s:%d %d"
                % (
                    stat.module[stat.module.rfind("/") + 1 :],
                    stat.name,
                    stat.lineno,
                    stat.ncall,
                )
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        type=str,
        choices=("profile", "trace", "coverage", "pdb", "yappi", "none"),
        nargs="?",
        default="none",
        help="The tracing/profiling case to test",
    )
    parser.add_argument(
        "-s",
        "--scheduler",
        type=str,
        choices=("greedy", "multithread", "event_based"),
        default="greedy",
        help="The scheduler to use",
    )
    args = parser.parse_args()

    if args.command == "profile":
        profile_main(args.scheduler)
    elif args.command == "trace":
        trace_main(args.scheduler)
    elif args.command == "coverage":
        coverage_main(args.scheduler)
    elif args.command == "pdb":
        pdb_main(args.scheduler)
    elif args.command == "yappi":
        yappi_main(args.scheduler)
    elif args.command == "none":
        main(args.scheduler)
