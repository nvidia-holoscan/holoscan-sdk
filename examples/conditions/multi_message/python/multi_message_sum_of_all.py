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
        op_output.emit(self.message, "out")


class SumOfAllThrottledRxOp(Operator):
    def __init__(self, fragment, *args, message="", **kwargs):
        self.message = message
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        # Using size argument to explicitly set the receiver message queue size for each input.
        spec.input("in1", size=IOSpec.IOSize(20))
        spec.input("in2", size=IOSpec.IOSize(20))
        spec.input("in3", size=IOSpec.IOSize(20))

        # Use `kMultiMessageAvailableTimeout`` to considers all three ports together. In this
        # "SumOfAll" mode, it only matters that `min_sum` messages have arrived across all the
        # ports that are listed in `input_port_names` below, but it does not matter which ports the
        # messages arrived on. The "execution_frequency" is set to 300ms, so the operator can run
        # once 300 ms has elapsed even if 20 messages have not arrived. Use
        # `ConditionType.MULTI_MESSAGE_AVAILABLE instead` if the timeout interval is not desired.
        spec.multi_port_condition(
            kind=ConditionType.MULTI_MESSAGE_AVAILABLE_TIMEOUT,
            execution_frequency="300ms",
            port_names=["in1", "in2", "in3"],
            sampling_mode="SumOfAll",
            min_sum=20,
        )

    def compute(self, op_input, op_output, context):
        msg1 = op_input.receive("in1")
        msg2 = op_input.receive("in2")
        msg3 = op_input.receive("in3")
        print(f"messages received on in1: {msg1}")
        print(f"messages received on in2: {msg2}")
        print(f"messages received on in3: {msg3}")


class MultiMessageThrottledApp(Application):
    def compose(self):
        period1 = PeriodicCondition(self, recess_period=datetime.timedelta(milliseconds=40))
        tx1 = StringTxOp(self, period1, message="tx1", name="tx1")

        period2 = PeriodicCondition(self, recess_period=datetime.timedelta(milliseconds=80))
        tx2 = StringTxOp(self, period2, message="tx2", name="tx2")

        period3 = PeriodicCondition(self, recess_period=datetime.timedelta(milliseconds=160))
        tx3 = StringTxOp(self, period3, message="tx3", name="tx3")

        multi_rx_timeout = SumOfAllThrottledRxOp(
            self, CountCondition(self, count=5), name="multi_rx_timeout"
        )

        # Connect the operators into the workflow
        self.add_flow(tx1, multi_rx_timeout, {("out", "in1")})
        self.add_flow(tx2, multi_rx_timeout, {("out", "in2")})
        self.add_flow(tx3, multi_rx_timeout, {("out", "in3")})


def main():
    app = MultiMessageThrottledApp()
    app.scheduler(EventBasedScheduler(app, worker_thread_number=4))
    app.run()


if __name__ == "__main__":
    main()
