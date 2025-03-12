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

import time

from holoscan.core import Application, Operator
from holoscan.schedulers import EventBasedScheduler


class SimpleOp(Operator):
    """Simple operator that logs its name and sleeps."""

    def compute(self, op_input, op_output, context):
        print(f"executing operator: {self.name}")
        time.sleep(1)


## Alternative way to define the operator
# from holoscan.decorator import create_op


# @create_op(op_param="self")
# def simple_op(self):
#     """Simple operator that logs its name and sleeps."""
#     print(f"executing operator: {self.name}")
#     time.sleep(1)


class ForkJoinExecutionApp(Application):
    def compose(self):
        # Define the operators
        node1 = SimpleOp(self, name="node1")
        node2 = SimpleOp(self, name="node2")
        node3 = SimpleOp(self, name="node3")
        node4 = SimpleOp(self, name="node4")
        node5 = SimpleOp(self, name="node5")
        node6 = SimpleOp(self, name="node6")
        node7 = SimpleOp(self, name="node7")
        node8 = SimpleOp(self, name="node8")

        # # If you use the decorator, you can use the following code:
        # node1 = simple_op(self, name="node1")
        # node2 = simple_op(self, name="node2")
        # node3 = simple_op(self, name="node3")
        # node4 = simple_op(self, name="node4")
        # node5 = simple_op(self, name="node5")
        # node6 = simple_op(self, name="node6")
        # node7 = simple_op(self, name="node7")
        # node8 = simple_op(self, name="node8")

        # Define the fork-join workflow
        #                <|start|>
        #                    |
        #                  node1
        #         /    /     |     \     \
        #      node2 node3 node4 node5 node6
        #        \     \     |     /     /
        #         \     \    |    /     /
        #          \     \   |   /     /
        #           \     \  |  /     /
        #                  node7
        #                    |
        #                  node8
        self.add_flow(self.start_op(), node1)
        self.add_flow(node1, node2)
        self.add_flow(node1, node3)
        self.add_flow(node1, node4)
        self.add_flow(node1, node5)
        self.add_flow(node1, node6)
        self.add_flow(node2, node7)
        self.add_flow(node3, node7)
        self.add_flow(node4, node7)
        self.add_flow(node5, node7)
        self.add_flow(node6, node7)
        self.add_flow(node7, node8)


def main():
    app = ForkJoinExecutionApp()
    scheduler = EventBasedScheduler(app, worker_thread_number=5, stop_on_deadlock=True)
    app.scheduler(scheduler)
    app.run()


if __name__ == "__main__":
    main()

# Expected output:
# (node2 to node6 are executed in parallel so the output is not deterministic)
#
# executing operator: node1
# executing operator: node2
# executing operator: node3
# executing operator: node4
# executing operator: node5
# executing operator: node6
# executing operator: node7
# executing operator: node8
