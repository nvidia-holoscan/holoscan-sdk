# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import PingRxOp, PingTxOp

# define custom Operators for use in the demo


class PingMxOp(Operator):
    """Example of an operator modifying data.

    This operator has 1 input and 1 output port:
        input:  "in"
        output: "out"

    The data from each input is multiplied by a user-defined value.

    """

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")
        spec.param("multiplier", 2)

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        print(f"Middle message value: {value}")

        # Multiply the values by the multiplier parameter
        value *= self.multiplier

        op_output.emit(value, "out")


# Now define a simple application using the operators defined above


class MyPingApp(Application):
    def compose(self):
        # Define the tx, mx, rx operators, allowing the tx operator to execute 10 times
        tx = PingTxOp(self, CountCondition(self, 10), name="tx")
        mx = PingMxOp(self, name="mx", multiplier=3)
        rx = PingRxOp(self, name="rx")

        # Define the workflow:  tx -> mx -> rx
        self.add_flow(tx, mx)
        self.add_flow(mx, rx)


if __name__ == "__main__":
    app = MyPingApp()
    app.run()
