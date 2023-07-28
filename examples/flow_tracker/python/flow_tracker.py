# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from holoscan.core import Application, Operator, OperatorSpec, Tracker

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
        output: "out"

    On each tick, it transmits a `ValueData` object
    """

    def __init__(self, fragment, *args, **kwargs):
        self.index = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        value = ValueData(self.index)
        self.index += 1
        op_output.emit(value, "out")


class PingMxOp(Operator):
    """Example of an operator modifying data.

    This operator has:
        input:  "in"
        output: "out"

    The data from each input is multiplied by a user-defined value.
    """

    def __init__(self, fragment, *args, **kwargs):
        self.count = 1

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")
        spec.param("multiplier", 2)

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        print(f"Middle message received (count: {self.count})")
        self.count += 1

        print(f"Middle message value: {value.data}")

        # Multiply the values by the multiplier parameter
        value.data *= self.multiplier

        op_output.emit(value, "out")


class PingRxOp(Operator):
    """Simple receiver operator.

    This operator has:
        input: "receivers"

    This is an example of a native operator that can dynamically have any
    number of inputs connected to is "receivers" port.
    """

    def __init__(self, fragment, *args, **kwargs):
        self.count = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.param("receivers", kind="receivers")

    def compute(self, op_input, op_output, context):
        values = op_input.receive("receivers")
        print(f"Rx message received (count: {self.count}, size: {len(values)})")
        self.count += 1

        print(values)
        print(f"Rx message value0: {values[0].data}")


# Now define a simple application using the operators defined above


class MyPingApp(Application):
    def compose(self):
        root1 = PingTxOp(self, CountCondition(self, 10), name="root1")
        root2 = PingTxOp(self, CountCondition(self, 10), name="root2")
        middle1 = PingMxOp(self, name="middle1", multiplier=2)
        middle2 = PingMxOp(self, name="middle2", multiplier=3)
        leaf1 = PingRxOp(self, name="leaf1")
        leaf2 = PingRxOp(self, name="leaf2")

        # Define the workflow
        self.add_flow(root1, middle1)
        self.add_flow(middle1, leaf1, {("out", "receivers")})
        self.add_flow(middle1, leaf2, {("out", "receivers")})

        self.add_flow(root2, middle2)
        self.add_flow(middle2, leaf2, {("out", "receivers")})


if __name__ == "__main__":
    app = MyPingApp()
    with Tracker(
        app, filename="logger.log", num_start_messages_to_skip=2, num_last_messages_to_discard=3
    ) as tracker:
        app.run()
        tracker.print()
