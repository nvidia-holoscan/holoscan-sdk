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


class SimpleOp(Operator):
    """Simple operator that prints its name and index."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = 0

    def compute(self, op_input, op_output, context):
        self.value += 1
        index = self.metadata.get("index", -1)
        print(f"I am here - {self.name} (index: {index})")


## Alternative way to define the operator
# from holoscan.decorator import create_op
#
# @create_op(op_param="self")
# def simple_op(self):
#     if not hasattr(self, "value"):
#         self.value = 0
#     self.value += 1
#     index = self.metadata.get("index", -1)
#     print(f"I am here - {self.name} (index: {index})")


# class DynamicFlowCallback:
#     """Callback class to handle dynamic flow control."""

#     def __init__(self, node2, node4):
#         self.index = 0
#         self.node2 = node2
#         self.node4 = node4

#     def __call__(self, op):
#         op.metadata.set("index", self.index)

#         if self.index < 3:
#             op.add_dynamic_flow(self.node2)
#             self.index += 1
#         else:
#             self.index = 0
#             op.add_dynamic_flow(self.node4)


class ForLoopExecutionApp(Application):
    def compose(self):
        # Define the operators
        node1 = SimpleOp(self, name="node1")
        node2 = SimpleOp(self, name="node2")
        node3 = SimpleOp(self, name="node3")
        node4 = SimpleOp(self, name="node4")

        ## If decorator is used, the operator is defined as follows:
        # node1 = simple_op(self, name="node1")
        # node2 = simple_op(self, name="node2")
        # node3 = simple_op(self, name="node3")
        # node4 = simple_op(self, name="node4")

        # Define the for-loop workflow
        #
        # Node Graph:
        #
        #     <|start|>
        #         |
        #       node1
        #       / ^  \
        #   node2 |  node4
        #     |   |
        #   node3 |
        #     |   |(loop)
        #     \__/
        self.add_flow(self.start_op(), node1)
        self.add_flow(node1, node2)
        self.add_flow(node2, node3)
        self.add_flow(node3, node1)
        self.add_flow(node1, node4)

        # As node1 operates in a loop, the following metadata policy will be (automatically)
        # internally configured to permit metadata updates during the application's execution:
        #   from holoscan.core import MetadataPolicy
        #   node1.metadata_policy = MetadataPolicy.UPDATE
        def dynamic_flow_callback(op):
            if not hasattr(dynamic_flow_callback, "index"):
                dynamic_flow_callback.index = 0

            op.metadata.set("index", dynamic_flow_callback.index)

            if dynamic_flow_callback.index < 3:
                op.add_dynamic_flow(node2)
                dynamic_flow_callback.index += 1
            else:
                dynamic_flow_callback.index = 0
                op.add_dynamic_flow(node4)

        self.set_dynamic_flows(node1, dynamic_flow_callback)

        # # Alternative way to set dynamic flows
        # # Create callback instance and set dynamic flows
        # self.set_dynamic_flows(node1, DynamicFlowCallback(node2, node4))


def main():
    app = ForLoopExecutionApp()
    app.run()


if __name__ == "__main__":
    main()

# Expected output:
# (node1 will loop 3 times through node2->node3 before going to node4)
#
# I am here - node1 (index: -1)
# I am here - node2 (index: 0)
# I am here - node3 (index: 0)
# I am here - node1 (index: 0)
# I am here - node2 (index: 1)
# I am here - node3 (index: 1)
# I am here - node1 (index: 1)
# I am here - node2 (index: 2)
# I am here - node3 (index: 2)
# I am here - node1 (index: 2)
# I am here - node4 (index: 3)
