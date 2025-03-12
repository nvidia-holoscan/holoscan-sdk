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


class PingRx(Operator):
    """Receiver operator that receives values."""

    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("input")
        if msg is not None:
            print(f"Received value {msg} - {self.name}")


## Alternative way to define the operator
# from holoscan.decorator import Input, create_op


# @create_op(op_param="self", outputs="output")
# def ping_tx(self):
#     """Transmitter operator that sends incrementing values."""
#     if not hasattr(self, "value"):
#         self.value = 0
#     self.value += 1
#     print(f"Sending value {self.value} - {self.name}")
#     return self.value


# @create_op(op_param="self", inputs=Input("input", arg_map="msg"))
# def ping_rx(self, msg):
#     """Receiver operator that receives values."""
#     if msg is not None:
#         print(f"Received value {msg} - {self.name}")


class ConditionalRoutingApp(Application):
    def compose(self):
        # Define the operators
        node1 = PingTx(self, CountCondition(self, count=2), name="node1")
        node2 = PingRx(self, name="node2")
        node3 = PingRx(self, name="node3")

        # # If decorator is used, the operator is defined as follows:
        # node1 = ping_tx(self, CountCondition(self, count=2), name="node1")
        # node2 = ping_rx(self, name="node2")
        # node3 = ping_rx(self, name="node3")

        # Define the conditional routing workflow
        #
        # Node Graph:
        #
        #             node1 (launch twice, emitting data to 'output' port)
        #            /     \
        #          node2 node3

        self.add_flow(node1, node2)
        self.add_flow(node1, node3)

        # # If you want to add all the next flows, you can use the following code:
        # self.set_dynamic_flows(node1, lambda op: op.add_dynamic_flow(op.next_flows))

        def dynamic_flow_callback(op):
            if op.value % 2 == 1:
                op.add_dynamic_flow(node2)
            else:
                op.add_dynamic_flow(node3)

        self.set_dynamic_flows(node1, dynamic_flow_callback)

        ## Simplified version of above code
        # self.set_dynamic_flows(
        #     node1,
        #     lambda op: op.add_dynamic_flow(node2)
        #     if op.value % 2 == 1
        #     else op.add_dynamic_flow(node3),
        # )

        ## This is another way to add dynamic flows based on the next operator name
        # def dynamic_flow_callback(op):
        #     node2_flow = op.find_flow_info(lambda flow: flow.next_operator.name == "node2")
        #     node3_flow = op.find_flow_info(lambda flow: flow.next_operator.name == "node3")

        #     # all_next_flows = op.find_all_flow_info(lambda flow: True)
        #     # print(f"All next flows: {[flow.next_operator.name for flow in all_next_flows]}")

        #     if op.value % 2 == 1:
        #         op.add_dynamic_flow(node2_flow)
        #     else:
        #         op.add_dynamic_flow(node3_flow)

        # self.set_dynamic_flows(node1, dynamic_flow_callback)


def main():
    app = ConditionalRoutingApp()
    app.run()


if __name__ == "__main__":
    main()

# Expected output:
#
# Sending value 1 - node1
# Received value 1 - node2
# Sending value 2 - node1
# Received value 2 - node3
