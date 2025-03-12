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

from holoscan.core import Application, Operator


class SimpleOp(Operator):
    """Simple operator that prints its name during compute."""

    def compute(self, op_input, op_output, context):
        print(f"I am here - {self.name}")


## Alternative way to define the operator
# from holoscan.decorator import create_op
#
# @create_op(op_param="self")
# def simple_op(self):
#     """Simple operator that prints its name during compute."""
#     print(f"I am here - {self.name}")


class SequentialExecutionApp(Application):
    def compose(self):
        # Define the operators
        node1 = SimpleOp(self, name="node1")
        node2 = SimpleOp(self, name="node2")
        node3 = SimpleOp(self, name="node3")

        ## If decorator is used, the operator is defined as follows:
        # node1 = simple_op(self, name="node1")
        # node2 = simple_op(self, name="node2")
        # node3 = simple_op(self, name="node3")

        # Define the sequential workflow
        self.add_flow(self.start_op(), node1)
        self.add_flow(node1, node2)
        self.add_flow(node2, node3)


def main():
    app = SequentialExecutionApp()
    app.run()


if __name__ == "__main__":
    main()

# Expected output:
#
# I am here - node1
# I am here - node2
# I am here - node3
