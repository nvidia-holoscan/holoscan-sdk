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
"""

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator


class SimpleOp(Operator):
    """Simple operator that prints its name."""

    def compute(self, op_input, op_output, context):
        print(f"I am here - {self.name}")


## Alternative way to define the operator
# from holoscan.decorator import Input, create_op


# @create_op(op_param="self")
# def simple_op(self):
#     print(f"I am here - {self.name}")


class PingTx(Operator):
    """Transmitter operator that sends incrementing values."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = 0

    def setup(self, spec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        self.value += 1
        print(f"Sending value {self.value} - {self.name}")
        op_output.emit(self.value, "output")


## Alternative way to define the operator
# @create_op(op_param="self", outputs="output")
# def ping_tx(self):
#     if not hasattr(self, "value"):
#         self.value = 0
#     self.value += 1
#     print(f"Sending value {self.value} - {self.name}")
#     return self.value


class PingRx(Operator):
    """Receiver operator that receives values."""

    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("input")
        if msg is not None:
            print(f"Received value {msg} - {self.name}")


## Alternative way to define the operator
# @create_op(op_param="self", inputs=Input("input", arg_map="msg"))
# def ping_rx(self, msg):
#     if msg is not None:
#         print(f"Received value {msg} - {self.name}")


class MixedFlowApp(Application):
    def compose(self):
        # Define the operators
        node1 = PingTx(self, CountCondition(self, count=2), name="node1")
        node2 = PingRx(self, name="node2")
        alarm = SimpleOp(self, name="alarm")
        node3 = PingRx(self, name="node3")

        # # If decorator is used, the operator is defined as follows:
        # node1 = ping_tx(self, CountCondition(self, count=2), name="node1")
        # node2 = ping_rx(self, name="node2")
        # alarm = simple_op(self, name="alarm")
        # node3 = ping_rx(self, name="node3")

        # Define the mixed flows workflow
        #
        # Node Graph:
        #
        #             node1 (from 'output' port)
        #         /     |     \
        #       node2 alarm  node3

        # The connections from `node1` to either `node2` or `node3` require an explicit port name
        # because the `node1` operator has two output ports: `output` and `__output_exec__`
        # (implicitly added by the framework when dynamic flows are used).
        # When connecting a specific output port to the input port of the next operator,
        # you must explicitly specify the port name.
        self.add_flow(node1, node2, {("output", "input")})
        self.add_flow(node1, alarm)
        self.add_flow(node1, node3, {("output", "input")})

        def dynamic_flow_callback(op):
            if op.value % 2 == 1:
                op.add_dynamic_flow("output", node2, "input")
            else:
                op.add_dynamic_flow("output", node3, "input")

            # Since the `node1` operator has three outgoing flows, we need to specify the output
            # port name explicitly to the `add_dynamic_flow()` function.
            #
            # `Operator.OUTPUT_EXEC_PORT_NAME` is the default execution output port name for
            # operators which would be used by Holoscan internally to connect the output of the
            # operator to the input of the next operator.
            #
            # There is `Operator.INPUT_EXEC_PORT_NAME` which is similar to `OUTPUT_EXEC_PORT_NAME`
            # but for the input port.
            #
            # Here we are using `OUTPUT_EXEC_PORT_NAME` to signal self to trigger the alarm
            # operator.
            op.add_dynamic_flow(Operator.OUTPUT_EXEC_PORT_NAME, alarm)

        self.set_dynamic_flows(node1, dynamic_flow_callback)


def main():
    app = MixedFlowApp()
    app.run()


if __name__ == "__main__":
    main()

# Expected output:
# (node1 alternates between sending to node2 and node3, while alarm is always executed)
#
# Sending value 1 - node1
# I am here - alarm
# Received value 1 - node2
# Sending value 2 - node1
# Received value 2 - node3
# I am here - alarm
