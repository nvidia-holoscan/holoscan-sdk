"""
 SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import multiprocessing
import time
from argparse import ArgumentParser

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler, MultiThreadScheduler


class PingTxOp(Operator):
    """Simple transmitter operator.

    This operator has:
        outputs: "out"

    On each tick, it transmits an integer on the "out" port. The transmitted value is incremented
    with each call to compute.
    """

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        op_output.emit(0, "out")


class DelayOp(Operator):
    """Example of an operator modifying data.

    This operator waits for a specified delay and then increments the received
    value by a user-specified integer increment.
    """

    def __init__(self, fragment, *args, delay=0.25, increment=1, **kwargs):
        self.delay = delay
        self.increment = increment

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out_name")
        spec.output("out_val")

    def compute(self, op_input, op_output, context):
        print(f"{self.name}: now waiting {self.delay:0.3f} s")
        time.sleep(self.delay)
        print(f"{self.name}: finished waiting")
        new_value = op_input.receive("in") + self.increment
        op_output.emit(self.name, "out_name")
        print(f"{self.name}: sending new value ({new_value})")
        op_output.emit(new_value, "out_val")


class PingRxOp(Operator):
    """Simple (multi)-receiver operator.

    This is an example of a native operator that can dynamically have any
    number of inputs connected to is "receivers" port.
    """

    def __init__(self, fragment, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.param("names", kind="receivers")
        spec.param("values", kind="receivers")

    def compute(self, op_input, op_output, context):
        # In this case, nothing will be printed until all messages have
        # been received.
        names = op_input.receive("names")
        values = op_input.receive("values")
        print(f"number of received names: {len(names)}")
        print(f"number of received values: {len(values)}")
        print(f"sum of received values: {sum(values)}")


# Now define a simple application using the operators defined above


class ParallelPingApp(Application):
    def __init__(self, *args, num_delays=8, delay=0.5, delay_step=0.1, **kwargs):
        self.num_delays = num_delays
        self.delay = delay
        self.delay_step = delay_step
        super().__init__(*args, **kwargs)

    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent.
        tx = PingTxOp(self, CountCondition(self, 1), name="tx")
        delay_ops = [
            DelayOp(self, delay=self.delay + self.delay_step * n, increment=n, name=f"delay{n:02d}")
            for n in range(self.num_delays)
        ]
        rx = PingRxOp(self, name="rx")
        for d in delay_ops:
            self.add_flow(tx, d)
            self.add_flow(d, rx, {("out_val", "values"), ("out_name", "names")})


def main(threads, num_delays, delay, delay_step, event_based):
    app = ParallelPingApp(num_delays=num_delays, delay=delay, delay_step=delay_step)
    if threads == 0:
        # Explicitly setting GreedyScheduler is not strictly required as it is the default.
        scheduler = GreedyScheduler(app, name="greedy_scheduler")
    else:
        scheduler_class = EventBasedScheduler if event_based else MultiThreadScheduler
        scheduler = scheduler_class(
            app,
            worker_thread_number=threads,
            stop_on_deadlock=True,
            stop_on_deadlock_timeout=500,
            name="multithread_scheduler",
        )
    app.scheduler(scheduler)
    tstart = time.time()
    app.run()
    duration = time.time() - tstart
    print(f"Total app runtime = {duration:0.3f} s")


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Parallel operator example")
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=-1,
        help=(
            "The number of threads to use for multi-threaded schedulers. Set this to 0 to use "
            "the default greedy scheduler instead. If set to -1, multiprocessing.cpu_count() "
            "threads will be used. To use the event-based scheduler instead of the default "
            "multi-thread scheduler, please specify --event_based."
        ),
    )
    parser.add_argument(
        "-n",
        "--num_delay_ops",
        type=int,
        default=32,
        help=(
            "The number of delay operators to launch. These delays will run in parallel (up to the "
            "specified number of threads)."
        ),
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=float,
        default=0.1,
        help=("The base amount of delay for each delay operator."),
    )
    parser.add_argument(
        "-s",
        "--delay_step",
        type=float,
        default=0.01,
        help=(
            "If nonzero, operators have variable delay. The jth delay operator"
            "will have a delay equal to `delay + j*delay_step` where j runs from "
            "0 to (num_delay_ops - 1)."
        ),
    )
    parser.add_argument(
        "--event_based",
        action="store_true",
        help=(
            "Sets the application to use the event-based scheduler instead of the default "
            "multi-thread scheduler when threads > 0."
        ),
    )

    args = parser.parse_args()
    if args.delay < 0:
        raise ValueError("delay must be non-negative")
    if args.delay_step < 0:
        raise ValueError("delay_step must be non-negative")
    if args.num_delay_ops < 1:
        raise ValueError("num_delay_ops must be >= 1")
    if args.threads < -1:
        raise ValueError("threads must be non-negative or -1 (for all threads)")
    elif args.threads == -1:
        # use up to maximum number of available threads
        args.threads = min(args.num_delay_ops, multiprocessing.cpu_count())

    main(
        threads=args.threads,
        num_delays=args.num_delay_ops,
        delay=args.delay,
        delay_step=args.delay_step,
        event_based=args.event_based,
    )
