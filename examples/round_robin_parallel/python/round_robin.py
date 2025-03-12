"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time
from argparse import ArgumentParser

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, ConditionType, Operator, OperatorSpec
from holoscan.operators import PingRxOp, PingTxOp
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler


class RoundRobinBroadcastOp(Operator):
    """Round-robin broadcast operator.

    A Holoscan operator that broadcasts input messages to multiple output ports in a round-robin
    fashion.

    The output ports will be named "output001", "output002", ..., ``f"output{num_broadcast:03d}"``.

    This operator receives messages on a single input port and broadcasts them to multiple output
    ports in a round-robin manner. The number of output ports is determined by the "num_broadcast"
    parameter.
    """

    def __init__(self, fragment, *args, num_broadcast=4, **kwargs):
        if num_broadcast < 1:
            raise RuntimeError("Must have num_broadcast >= 1 (at least one output port).")
        self.num_broadcast = num_broadcast
        self.port_index = 0
        super().__init__(fragment, *args, **kwargs)

    def output_name(self, index: int):
        """Return the name of the output port name associated with a given linear port index.

        Parameters
        ----------
        index : int
            Valid index values are in the range [0, self.num_broadcast).

        Returns
        -------
        name : str
            The name of the output port at the given `index`.
        """
        return f"output{index + 1:03d}"

    def setup(self, spec: OperatorSpec):
        spec.input("input")

        # create output ports output001, output002, ...
        for i in range(self.num_broadcast):
            port_name = self.output_name(i)
            spec.output(port_name)

    def compute(self, op_input, op_output, context):
        message = op_input.receive("input")
        op_output.emit(message, self.output_name(self.port_index))
        self.port_index = (self.port_index + 1) % self.num_broadcast


class SlowOp(Operator):
    """Example of an operator modifying data.

    This operator waits for a specified delay and then increments the received
    value by a user-specified integer increment.

    In a real-world application the delay would be some computation that takes
    a substantial amount of time but may not fully saturate CPU or GPU
    resources such that acceleration could be achieved by running multiple
    copies of the operator in parallel.
    """

    def __init__(self, fragment, *args, delay=0.25, increment=1, silent=False, **kwargs):
        self.delay = delay
        self.increment = increment
        self.silent = silent

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        new_value = op_input.receive("in") + self.increment

        if self.delay > 0:
            if not self.silent:
                print(f"{self.name}: now waiting {self.delay:0.3f} s")
            time.sleep(self.delay)
            if not self.silent:
                print(f"{self.name}: finished waiting")
        if not self.silent:
            print(f"{self.name}: sending new value ({new_value})")
        op_output.emit(new_value, "out")


class GatherOneOp(Operator):
    """Round-robin gather operator.

    A Holoscan operator that has "num_gather" input ports and a single output port. This operator
    only checks a specific input port for messages on any given `compute` call. Which input port is
    checked varies across compute calls in a round-robin fashion.

    For the way add_flow calls are made in `RoundRobinApp`, this ensures that the original order of
    frames produced by `PingTxOp` is preserved.

    The input ports will be named "input001", "input002", ...,
    `fmt::format("input{:03d}", num_gather)`.
    """

    def __init__(self, fragment, *args, num_gather=4, **kwargs):
        self.num_gather = num_gather
        self.port_index = 0
        super().__init__(fragment, *args, **kwargs)

    def input_name(self, index: int):
        """Return the name of the input port name associated with a given linear port index.

        Parameters
        ----------
        index : int
            Valid index values are in the range [0, self.num_gather).

        Returns
        -------
        name : str
            The name of the input port at the given `index`.
        """
        return f"input{index + 1:03d}"

    def setup(self, spec: OperatorSpec):
        # create input ports input001, input002, ...
        for i in range(self.num_gather):
            port_name = self.input_name(i)
            spec.input(port_name)

        # only call compute if a message has arrived on one of the input ports
        spec.multi_port_condition(
            kind=ConditionType.MULTI_MESSAGE_AVAILABLE,
            port_names=[self.input_name(i) for i in range(self.num_gather)],
            sampling_mode="SumOfAll",
            min_sum=1,
        )

        spec.output("output")

    def compute(self, op_input, op_output, context):
        message = op_input.receive(self.input_name(self.port_index))
        # if message is not available on the current port index, return early
        if message:
            self.port_index = (self.port_index + 1) % self.num_gather
            # output the message and increment the port number
            op_output.emit(message, "output")


# Now define a simple application using the operators defined above
class RoundRobinApp(Application):
    """This application illustrates round-robin dispatch to multiple branches.

    Please see ../README.md for a detailed description and diagram of the operators involved.
    """

    def __init__(self, *args, num_broadcast=4, silent=False, delay=1 / 15, **kwargs):
        self.num_broadcast = num_broadcast
        # parameters for SlowOp
        self.delay = delay
        self.silent = silent
        super().__init__(*args, **kwargs)

    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent and a
        # PeriodicCondition to control how often messages are sent.
        source_rate_hz = 60  # messages sent per second
        period_source_ns = int(1e9 / source_rate_hz)  # period in nanoseconds
        tx = PingTxOp(
            self,
            CountCondition(self, 100),
            PeriodicCondition(self, recess_period=period_source_ns),
            start=1,
            increment=1,
            name="tx",
        )

        round_robin = RoundRobinBroadcastOp(
            self, num_broadcast=self.num_broadcast, name="round_robin"
        )

        slow_ops = [
            SlowOp(
                self,
                delay=self.delay,
                increment=0,
                silent=self.silent,
                name=f"delay{n:02d}",
            )
            for n in range(self.num_broadcast)
        ]

        gather = GatherOneOp(self, num_gather=self.num_broadcast, name="gather")

        rx = PingRxOp(self, name="rx")

        self.add_flow(tx, round_robin)
        for i, slow_op in enumerate(slow_ops):
            self.add_flow(round_robin, slow_op, {(round_robin.output_name(i), "in")})
            self.add_flow(slow_op, gather, {("out", gather.input_name(i))})
        self.add_flow(gather, rx)


def main(threads):
    app = RoundRobinApp()
    if threads == 0:
        # Explicitly setting GreedyScheduler is not strictly required as it is the default.
        scheduler = GreedyScheduler(app, name="greedy_scheduler")
    else:
        scheduler = EventBasedScheduler(
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
    parser = ArgumentParser(description="Multi-rate pipeline example")
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=8,
        help=(
            "The number of threads to use for the event-based scheduler. Set this to 0 to use "
            "the single-threaded greedy scheduler instead."
        ),
    )

    args = parser.parse_args()
    if args.threads < 0:
        raise ValueError("threads must be non-negative")

    main(threads=args.threads)
