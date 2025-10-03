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

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, ConditionType, IOSpec, Operator, OperatorSpec


class PingTxOp(Operator):
    """Simple transmitter operator.

    On each tick, it transmits an integer to the "out" port.

    **==Named Outputs==**

        out : int
            An index value that increments by one on each call to `compute`. The starting value is
            1.
    """

    def __init__(self, fragment, *args, **kwargs):
        self.index = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # we can retrieve the scheduler used for this application via it's fragment
        scheduler = self.fragment.scheduler()

        # The scheduler's clock is available as a parameter.
        # The clock object has methods to retrieve timestamps.
        clock = scheduler.clock

        print(f"Sending message: {self.index}")

        # Note that we must set acq_timestamp so that the timestamp required by
        # the ExpiringMessageAvailableCondition on the downstream operator will
        # be found.
        op_output.emit(self.index, "out", acq_timestamp=clock.timestamp())
        self.index += 1


class PingRxOp(Operator):
    """Simple receiver operator.

    This is an example of a native operator with one input port.
    On each tick, it receives up to 5 batches of messages from the "in" port.
    If 5 messages are not received within 1 second, the operator will be triggered to process.

    **==Named Inputs==**

        in : any
            A received value.
    """

    def __init__(self, fragment, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=5,
            policy=1,
        ).condition(
            # Set the enum value corresponding to ExpiringMessageAvailableCondition
            ConditionType.EXPIRING_MESSAGE_AVAILABLE,
            max_batch_size=5,
            max_delay_ns=1_000_000_000,
        )

    def compute(self, op_input, op_output, context):
        message = op_input.receive("in")
        print("PingRxOp.compute() called")
        while message:
            print(f"ExpiringMessageAvailable ping: {message}")
            message = op_input.receive("in")

        # can get the timestamp emitted by PingTxOp if it is needed
        # (must be called after the receive call for this port)
        timestamp = op_input.get_acquisition_timestamp("in")
        assert timestamp is not None
        assert isinstance(timestamp, int)


# Now define a simple application using the operators defined above


class MyPingApp(Application):
    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent.
        # PeriodicCondition is used so that each subsequent message is
        # sent only after a period of 10 milliseconds has elapsed.
        tx = PingTxOp(self, CountCondition(self, 8), PeriodicCondition(self, 10_000_000), name="tx")
        rx = PingRxOp(self, name="rx")

        # Connect the operators into the workflow:  tx -> rx
        self.add_flow(tx, rx)


def main():
    app = MyPingApp()
    app.run()


if __name__ == "__main__":
    main()
