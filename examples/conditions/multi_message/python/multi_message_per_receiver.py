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

import datetime

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, ConditionType, IOSpec, Operator
from holoscan.schedulers import EventBasedScheduler

# Now define a simple application using the operators defined above


class StringTxOp(Operator):
    def __init__(self, fragment, *args, message="", **kwargs):
        self.message = message
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        print(f"{self.name}: sending message")
        op_output.emit(self.message, "out")


class PerReceiverRxOp(Operator):
    def __init__(self, fragment, *args, message="", **kwargs):
        self.message = message
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        # Using size argument to explicitly set the receiver message queue size for each input.
        spec.input("in1")
        spec.input("in2", size=IOSpec.IOSize(2))
        spec.input("in3")

        # Configure a MultiMessageAvailableCondition in "PerReceiver" mode so the operator will run
        # only when 1, 2 and 1 messages have arrived on ports "in1", "in2" and "in3", respectively.
        # This per-receiver mode is equivalent to putting a MessageAvailable condition on each input
        # individually.
        spec.multi_port_condition(
            kind=ConditionType.MULTI_MESSAGE_AVAILABLE,
            port_names=["in1", "in2", "in3"],
            sampling_mode="PerReceiver",
            min_sizes=[1, 2, 1],
        )

    def compute(self, op_input, op_output, context):
        msg1 = op_input.receive("in1")
        msg2 = op_input.receive("in2")
        msg3 = op_input.receive("in3")
        print(f"message received on in1: {msg1}")
        print(f"first message received on in2: {msg2[0]}")
        print(f"second message received on in2: {msg2[1]}")
        print(f"message received on in3: {msg3}")


class MultiMessageApp(Application):
    def compose(self):
        period1 = PeriodicCondition(self, recess_period=datetime.timedelta(milliseconds=50))
        tx1 = StringTxOp(self, period1, message="Hello from tx1", name="tx1")

        period2 = PeriodicCondition(self, recess_period=datetime.timedelta(milliseconds=25))
        tx2 = StringTxOp(self, period2, message="Hello from tx2", name="tx2")

        period3 = PeriodicCondition(self, recess_period=datetime.timedelta(milliseconds=100))
        tx3 = StringTxOp(self, period3, message="Hello from tx3", name="tx3")

        multi_rx = PerReceiverRxOp(self, CountCondition(self, count=4), name="multi_rx")

        # Connect the operators into the workflow
        self.add_flow(tx1, multi_rx, {("out", "in1")})
        self.add_flow(tx2, multi_rx, {("out", "in2")})
        self.add_flow(tx3, multi_rx, {("out", "in3")})


def main():
    app = MultiMessageApp()
    app.scheduler(EventBasedScheduler(app, worker_thread_number=4))
    app.run()


if __name__ == "__main__":
    main()
