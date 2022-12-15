# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.logger import load_env_log_level

# define a custom class to represent data used in the app


class ValueData:
    """Example of a custom Python class"""

    def __init__(self, value):
        self.data = value

    def __repr__(self):
        return f"ValueData({self.data})"

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)


# define custom Operators for use in the demo


class PingTxOp(Operator):
    """Simple transmitter operator.

    This operator has:
        outputs: "out1", "out2"

    On each tick, it transmits a `ValueData` object at each port. The
    transmitted values are even on port1 and odd on port2 and increment with
    each call to compute.
    """

    def __init__(self, *args, **kwargs):
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out1")
        spec.output("out2")

    def compute(self, op_input, op_output, context):
        value1 = ValueData(self.index)
        self.index += 1
        op_output.emit(value1, "out1")

        value2 = ValueData(self.index)
        self.index += 1
        op_output.emit(value2, "out2")


class PingMiddleOp(Operator):
    """Example of an operator modifying data.

    This operator has:
        inputs:  "in1", "in2"
        outputs: "out1", "out2"

    The data from each input is multiplied by a user-defined value.

    In this demo, the `multiplier` parameter value is read from a "ping.yaml"
    configuration file (near the bottom of this script), overriding the default
    defined in the setup() method below.
    """

    def __init__(self, *args, **kwargs):
        # If `self.multiplier` is set here (e.g., `self.multiplier = 4`), then
        # the default value by `param()` in `setup()` will be ignored.
        # (you can just call `spec.param("multiplier")` in `setup()` to use the
        # default value)
        #
        # self.multiplier = 4
        self.count = 1

        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in1")
        spec.input("in2")
        spec.output("out1")
        spec.output("out2")
        spec.param("multiplier", 2)

    def compute(self, op_input, op_output, context):
        value1 = op_input.receive("in1")
        value2 = op_input.receive("in2")
        print(f"Middle message received (count: {self.count})")
        self.count += 1

        print(f"Middle message value1: {value1.data}")
        print(f"Middle message value2: {value2.data}")

        # Multiply the values by the multiplier parameter
        value1.data *= self.multiplier
        value2.data *= self.multiplier

        op_output.emit(value1, "out1")
        op_output.emit(value2, "out2")


class PingRxOp(Operator):
    """Simple receiver operator.

    This operator has:
        input: "receivers"

    This is an example of a native operator that can dynamically have any
    number of inputs connected to is "receivers" port.
    """

    def __init__(self, *args, **kwargs):
        self.count = 1
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.param("receivers", kind="receivers")

    def compute(self, op_input, op_output, context):
        values = op_input.receive("receivers")
        print(f"Rx message received (count: {self.count}, size: {len(values)})")
        self.count += 1
        print(f"Rx message value1: {values[0].data}")
        print(f"Rx message value2: {values[1].data}")


# Now define a simple application using the operators defined above


class MyPingApp(Application):
    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent.
        tx = PingTxOp(self, CountCondition(self, 10), name="tx")
        mx = PingMiddleOp(self, self.from_config("mx"), name="mx")
        rx = PingRxOp(self, name="rx")

        # Connect the operators into the workflow:  tx -> mx -> rx
        self.add_flow(tx, mx, {("out1", "in1"), ("out2", "in2")})
        self.add_flow(mx, rx, {("out1", "receivers"), ("out2", "receivers")})


if __name__ == "__main__":
    load_env_log_level()
    app = MyPingApp()
    app.config(os.path.join(os.path.dirname(__file__), "ping.yaml"))
    app.run()
