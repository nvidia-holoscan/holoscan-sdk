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

from holoscan.core import Application, Operator


class PingTxOp(Operator):
    """Transmitter operator that sends incrementing values."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = 0

    def setup(self, spec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        self.value += 1
        print(f"{self.name} - Sending value {self.value}")
        op_output.emit(self.value, "output")


## Alternative way to define the operator
# from holoscan.decorator import Input, create_op


# @create_op(op_param="self", outputs="output")
# def ping_tx_op(self):
#     if not hasattr(self, "value"):
#         self.value = 0
#     self.value += 1
#     print(f"{self.name} - Sending value {self.value}")
#     return self.value


class PingMxOp(Operator):
    """Middle operator that receives and forwards values."""

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("input")
        print(f"{self.name} - Received value {value}")
        op_output.emit(value, "output")


## Alternative way to define the operator
# @create_op(op_param="self", inputs=Input("input", arg_map="value"), outputs="output")
# def ping_mx_op(self, value):
#     print(f"{self.name} - Received value {value}")
#     return value


class PingRxOp(Operator):
    """Receiver operator that receives values."""

    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("input")
        print(f"{self.name} - Received value {value}")


## Alternative way to define the operator
# @create_op(op_param="self", inputs=Input("input", arg_map="value"))
# def ping_rx_op(self, value):
#     print(f"{self.name} - Received value {value}")


class StreamExecutionApp(Application):
    def compose(self):
        # Define the operators
        # You can uncomment the following lines to add a periodic condition to the node1 operator
        #   from holoscan.conditions import PeriodicCondition
        #   node1 = PingTxOp(self, PeriodicCondition(self, 0.2), name="node1")
        node1 = PingTxOp(self, name="node1")
        node2 = PingMxOp(self, name="node2")
        node3 = PingMxOp(self, name="node3")
        node4 = PingRxOp(self, name="node4")

        ## If decorator is used, the operator is defined as follows:
        # node1 = ping_tx_op(self, name="node1")
        # node2 = ping_mx_op(self, name="node2")
        # node3 = ping_mx_op(self, name="node3")
        # node4 = ping_rx_op(self, name="node4")

        # Define the streaming workflow
        #
        # Node Graph:
        #                   (cycle)
        #                   -------
        #                   \     /
        #     <|start|>  ->  node1  ->  node2  ->  node3  ->  node4
        #             (triggered 5 times)
        self.add_flow(self.start_op(), node1)
        self.add_flow(node1, node1)
        # The following line requires an explicit port name because the
        # `node1` operator has two output ports: `output` and `__output_exec__` (implicitly added
        #  by the framework when add_dynamic_flow is used).
        # When connecting a specific output port to the input port of the next operator,
        # you must explicitly specify the port name.
        self.add_flow(node1, node2, {("output", "input")})
        self.add_flow(node2, node3)
        self.add_flow(node3, node4)

        def dynamic_flow_callback(op):
            if not hasattr(dynamic_flow_callback, "iteration"):
                dynamic_flow_callback.iteration = 0
            dynamic_flow_callback.iteration += 1
            print(f"#iteration: {dynamic_flow_callback.iteration}")

            if dynamic_flow_callback.iteration <= 5:
                op.add_dynamic_flow("output", node2)
                # Signal self to trigger the next iteration.
                # Since the `node1` operator has two outgoing flows, we need to specify the output
                # port name explicitly to the `add_dynamic_flow()` function.
                #
                # `Operator.OUTPUT_EXEC_PORT_NAME` is the default execution output port name for
                # operators which would be used by Holoscan internally to connect the output of the
                # operator to the input of the next operator.
                #
                # There is `Operator.INPUT_EXEC_PORT_NAME` which is similar to
                # `OUTPUT_EXEC_PORT_NAME` but for the input port.
                #
                # Here we are using `OUTPUT_EXEC_PORT_NAME` to signal the operator to trigger
                # itself in the next iteration.
                op.add_dynamic_flow(Operator.OUTPUT_EXEC_PORT_NAME, op)
            else:
                dynamic_flow_callback.iteration = 0

        self.set_dynamic_flows(node1, dynamic_flow_callback)


def main():
    app = StreamExecutionApp()
    app.run()


if __name__ == "__main__":
    main()

# Expected output:
#
# node1 - Sending value 1
# #iteration: 1
# node2 - Received value 1
# node3 - Received value 1
# node4 - Received value 1
# node1 - Sending value 2
# #iteration: 2
# node2 - Received value 2
# node3 - Received value 2
# node4 - Received value 2
# node1 - Sending value 3
# #iteration: 3
# node2 - Received value 3
# node3 - Received value 3
# node4 - Received value 3
# node1 - Sending value 4
# #iteration: 4
# node2 - Received value 4
# node3 - Received value 4
# node4 - Received value 4
# node1 - Sending value 5
# #iteration: 5
# node2 - Received value 5
# node3 - Received value 5
# node4 - Received value 5
# node1 - Sending value 6
# #iteration: 6
