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

import datetime

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, Operator
from holoscan.operators import PingTxOp


# Now define a simple application using the operators defined above
class MultiRxOrOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.count = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        spec.input("in1")
        spec.input("in2")

        # configure Operator to execute if an input is on "in1" OR "in2"
        # (without this, the default is "in1" AND "in2")
        spec.or_combine_port_conditions(("in1", "in2"))

    def compute(self, op_input, op_output, context):
        in_value1 = op_input.receive("in1")
        if in_value1:
            print(f"count {self.count}, message received on in1")
        in_value2 = op_input.receive("in2")
        if in_value2:
            print(f"count {self.count}, message received on in2")
        self.count += 1


class OrCombinerApp(Application):
    def compose(self):
        tx1 = PingTxOp(
            self,
            CountCondition(self, name="count", count=10),
            PeriodicCondition(self, recess_period=datetime.timedelta(milliseconds=30)),
            name="tx1",
        )
        tx2 = PingTxOp(
            self,
            CountCondition(self, name="count", count=10),
            PeriodicCondition(self, recess_period=datetime.timedelta(milliseconds=80)),
            name="tx2",
        )
        rx_or_combined = MultiRxOrOp(self, name="rx_or")

        # Connect the operators into the workflow
        self.add_flow(tx1, rx_or_combined, {("out", "in1")})
        self.add_flow(tx2, rx_or_combined, {("out", "in2")})


def main():
    app = OrCombinerApp()
    app.run()


if __name__ == "__main__":
    main()
