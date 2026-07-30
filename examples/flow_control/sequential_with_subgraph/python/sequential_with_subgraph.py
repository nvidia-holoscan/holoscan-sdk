# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from holoscan.core import Application, Operator, Subgraph


class SimpleOp(Operator):
    """Simple operator that prints its name"""

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def compute(self, op_input, op_output, context):
        print(f"I am here - {self.name}")


class SequentialSubgraph(Subgraph):
    """Subgraph containing a sequential execution flow"""

    def __init__(self, fragment, name):
        super().__init__(fragment, name)

    def compose(self):
        # Define the operators inside the subgraph
        node2 = SimpleOp(self, name="node2")
        node3 = SimpleOp(self, name="node3")

        # Define the sequential workflow inside the subgraph
        self.add_flow(node2, node3)

        # Expose node2's input execution port as the subgraph's input execution interface port
        self.add_input_exec_interface_port("exec_in", node2)

        # Expose node3's output execution port as the subgraph's output execution interface port
        self.add_output_exec_interface_port("exec_out", node3)


class SequentialWithSubgraphApp(Application):
    def compose(self):
        # Define operators outside the subgraph
        node1 = SimpleOp(self, name="node1")
        node4 = SimpleOp(self, name="node4")

        # Create the subgraph containing node2 and node3
        sequential_sg = SequentialSubgraph(self, name="sequential_sg")

        # Define the sequential workflow using control flow
        # start_op() -> node1 -> sequential_sg (node2 -> node3) -> node4
        self.add_flow(self.start_op(), node1)
        self.add_flow(node1, sequential_sg)  # Auto-resolves to exec_in
        self.add_flow(sequential_sg, node4)  # Auto-resolves to exec_out


if __name__ == "__main__":
    app = SequentialWithSubgraphApp()
    app.run()

# Expected output:
#
# I am here - node1
# I am here - node2
# I am here - node3
# I am here - node4
