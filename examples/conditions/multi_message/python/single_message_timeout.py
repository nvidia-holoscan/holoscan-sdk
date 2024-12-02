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
from holoscan.schedulers import GreedyScheduler

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


class RxTimeoutOp(Operator):
    def __init__(self, fragment, *args, message="", **kwargs):
        self.message = message
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        # Set a condition to allow execution once 5 messages have arrived or at least 250 ms has
        # elapsed since the prior time operator::compute was called.
        spec.input("in", size=IOSpec.IOSize(5)).condition(
            ConditionType.MULTI_MESSAGE_AVAILABLE_TIMEOUT, execution_frequency="250ms", min_sum=5
        )

    def compute(self, op_input, op_output, context):
        messages = op_input.receive("in")
        print(f"{len(messages)} messages received on in: {messages}")


class RxTimeoutApp(Application):
    def compose(self):
        tx = StringTxOp(
            self,
            PeriodicCondition(
                self, recess_period=datetime.timedelta(milliseconds=100), name="tx_period"
            ),
            CountCondition(self, count=18, name="tx_count"),
            message="tx",
            name="tx",
        )
        rx_timeout = RxTimeoutOp(self, name="rx_timeout")

        # Connect the operators into the workflow
        self.add_flow(tx, rx_timeout)


def main():
    app = RxTimeoutApp()
    # Add a timeout slightly less than the execution_frequency so any final messages have time to
    # arrive after tx stops calling compute. If the deadlock timeout here is > execution_frequency
    # than the receive operator will continue to call compute indefinitely with 0 messages at the
    # execution frequency.
    app.scheduler(GreedyScheduler(app, stop_on_deadlock_timeout=245))
    app.run()


if __name__ == "__main__":
    main()
