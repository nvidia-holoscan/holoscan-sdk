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
    """Simple operator that prints its name during compute."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = 0

    def compute(self, op_input, op_output, context):
        self.value += 1
        print(f"I am here - {self.name}")


## Alternative way to define the operator
# from holoscan.decorator import create_op
#
# @create_op(op_param="self")
# def simple_op(self):
#     if not hasattr(self, "value"):
#         self.value = 0
#     self.value += 1
#     print(f"I am here - {self.name} (value: {self.value})")


class ConditionalExecutionApp(Application):
    def compose(self):
        # Define the operators
        node1 = SimpleOp(self, CountCondition(self, count=2), name="node1")
        node2 = SimpleOp(self, name="node2")
        node3 = SimpleOp(self, name="node3")
        node4 = SimpleOp(self, name="node4")
        node5 = SimpleOp(self, name="node5")

        ## If decorator is used, the operator is defined as follows:
        # node1 = simple_op(self, CountCondition(self, count=2), name="node1")
        # node2 = simple_op(self, name="node2")
        # node3 = simple_op(self, name="node3")
        # node4 = simple_op(self, name="node4")
        # node5 = simple_op(self, name="node5")

        # Define the conditional workflow
        #
        # Node Graph:
        #
        #       node1 (launch twice)
        #       /   \
        #   node2   node4
        #     |       |
        #   node3   node5

        self.add_flow(node1, node2)
        self.add_flow(node2, node3)
        self.add_flow(node1, node4)
        self.add_flow(node4, node5)

        # # If you want to add all the next flows, you can use the following code:
        # self.set_dynamic_flows(node1, lambda op: op.add_dynamic_flow(op.next_flows))

        def dynamic_flow_callback(op):
            if op.value % 2 == 1:
                op.add_dynamic_flow(node2)
            else:
                op.add_dynamic_flow(node4)

        self.set_dynamic_flows(node1, dynamic_flow_callback)

        ## Simplified version of above code
        # self.set_dynamic_flows(
        #     node1,
        #     lambda op: op.add_dynamic_flow(node2)
        #     if op.value % 2 == 1
        #     else op.add_dynamic_flow(node4),
        # )

        ## This is another way to add dynamic flows based on the next operator name
        # def dynamic_flow_callback(op):
        #     node2_flow = op.find_flow_info(lambda flow: flow.next_operator.name == "node2")
        #     node4_flow = op.find_flow_info(lambda flow: flow.next_operator.name == "node4")

        #     # all_next_flows = op.find_all_flow_info(lambda flow: True)
        #     # print(f"All next flows: {[flow.next_operator.name for flow in all_next_flows]}")

        #     if op.value % 2 == 1:
        #         op.add_dynamic_flow(node2_flow)
        #     else:
        #         op.add_dynamic_flow(node4_flow)

        # self.set_dynamic_flows(node1, dynamic_flow_callback)


def main():
    app = ConditionalExecutionApp()
    app.run()


if __name__ == "__main__":
    main()

# Expected output:
#
# I am here - node1
# I am here - node2
# I am here - node3
# I am here - node1
# I am here - node4
# I am here - node5
