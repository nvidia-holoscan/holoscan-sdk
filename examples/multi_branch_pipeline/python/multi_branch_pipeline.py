"""
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from holoscan.core import Application, ConditionType, IOSpec, Operator, OperatorSpec
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler, MultiThreadScheduler


class PingTxOp(Operator):
    """Simple transmitter operator.

    This operator has:
        outputs: "out"

    On each tick, it transmits an integer on the "out" port. The value on the first call to
    compute is equal to `initial_value` and it increments by `increment` on each subsequent
    call.
    """

    def __init__(self, fragment, *args, initial_value=0, increment=0, **kwargs):
        self.count = 0
        self.initial_value = initial_value
        self.increment = increment

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        # Note: Setting ConditionType.NONE overrides the default of
        #   ConditionType.DOWNSTREAM_MESSAGE_AFFORDABLE. This means that the operator will be
        #   triggered regardless of whether any operators connected downstream have space in their
        #   queues.
        spec.output("out").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        value = self.initial_value + self.count * self.increment
        op_output.emit(value, "out")
        self.count += 1


class IncrementOp(Operator):
    """Add a fixed value to the input and transmit the result."""

    def __init__(self, fragment, *args, increment=0, **kwargs):
        self.increment = increment

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Setup the input and output ports with custom settings on the input port.

        Notes
        -----
        For policy:
          - IOSpec.QueuePolicy.POP = pop the oldest value in favor of the new one when the queue
            is full
          - IOSpec.QueuePolicy.REJECT = reject the new value when the queue is full
          - IOSpec.QueuePolicy.FAULT = fault if queue is full (default)

        For capacity:
          When capacity > 1, even once messages stop arriving, this entity will continue to
          call ``compute`` for each remaining item in the queue.

        The ``condition`` method call here is the same as the default setting, and is shown
        only for completeness. `min_size` = 1 means that this operator will not call compute
        unless there is at least one message in the queue.

        One could also set the receiver's capacity and policy via the connector method:

            .connector(
                IOSpec.ConnectorType.DOUBLE_BUFFER,
                capacity=1,
                policy=1,  # 1 = reject
            )

        but that is less flexible as `IOSpec::ConnectorType::kDoubleBuffer` is appropriate for
        within-fragment connections, but will not work if the operator was connected to a
        different fragment. By passing the capacity and policy as arguments to the
        `OperatorSpec::input` method, the SDK can still select the appropriate default receiver
        type depending on whether the connection is within-fragment or across fragments (or
        whether an annotated variants of the receiver class is needed for data flow tracking).

        Note that if policy was not specific in the `input` call as done here, it would also be
        possible for an application author to set the `policy` after the operator is constructed
        by using the `Operator.queue_policy` property (see commented code in
        `MultiRateApp.compose` below).
        """
        spec.input("in", policy=IOSpec.QueuePolicy.REJECT).condition(
            ConditionType.MESSAGE_AVAILABLE, min_size=1, front_stage_max_size=1
        )

        spec.output("out")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        new_value = value + self.increment
        op_output.emit(new_value, "out")


class PingRxOp(Operator):
    """Simple (multi)-receiver operator.

    This is an example of a native operator with one input port.
    On each tick, it receives an integer from the "in" port.
    """

    def __init__(self, fragment, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        # In this case, nothing will be printed until all messages have
        # been received.
        value = op_input.receive("in")
        print(f"receiver '{self.name}' received value: {value}")


# Now define a simple application using the operators defined above
class MultiRateApp(Application):
    """This application has a single transmitter connected to two parallel branches

    The geometry of the application is as shown below:

       increment1--rx1
      /
    tx
      \
       increment2--rx2

    The top branch is forced via a PeriodicCondition to run at a slower rate than
    the source. It is currently configured to discard any extra messages that arrive
    at increment1 before it is ready to execute again, but different behavior could be
    achieved via other settings to policy and/or queue sizes.

    """

    def __init__(self, *args, **kwargs):
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
            initial_value=0,
            increment=1,
            name="tx",
        )

        # first branch will have a periodic condition so it can't run faster than 5 Hz
        branch1_hz = 5
        period_ns1 = int(1e9 / branch1_hz)
        increment1 = IncrementOp(
            self,
            PeriodicCondition(self, recess_period=period_ns1),
            name="increment1",
        )
        # could set the queue policy here if it wasn't already set to REJECT in IncrementOp.start
        # increment1.queue_policy(
        #     "in", port_type=IOSpec.IOType.INPUT, policy=IOSpec.QueuePolicy.REJECT
        # )

        rx1 = PingRxOp(self, name="rx1")
        self.add_flow(tx, increment1)
        self.add_flow(increment1, rx1)

        # second branch does not have a periodic condition so will tick on every message sent by tx
        increment2 = IncrementOp(
            self,
            name="increment2",
        )
        # could set the queue policy here if it wasn't already set to REJECT in IncrementOp.start
        # increment2.queue_policy(
        #     "in", port_type=IOSpec.IOType.INPUT, policy=IOSpec.QueuePolicy.REJECT
        # )

        rx2 = PingRxOp(self, name="rx2")
        self.add_flow(tx, increment2)
        self.add_flow(increment2, rx2)


def main(threads, event_based):
    app = MultiRateApp()
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
    parser = ArgumentParser(description="Multi-rate pipeline example")
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=5,
        help=(
            "The number of threads to use for multi-threaded schedulers. Set this to 0 to use "
            "the default greedy scheduler instead. To use the event-based scheduler instead of "
            "the default multi-thread scheduler, please specify --event_based."
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
    if args.threads < 0:
        raise ValueError("threads must be non-negative")

    main(threads=args.threads, event_based=args.event_based)
