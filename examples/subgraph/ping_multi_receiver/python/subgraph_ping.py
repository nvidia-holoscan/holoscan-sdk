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

from holoscan.conditions import CountCondition
from holoscan.core import Application, IOSpec, Operator, OperatorSpec, Subgraph
from holoscan.operators import PingRxOp, PingTxOp


class ForwardingOp(Operator):
    """A simple pass-through operator that forwards any received message to its output port."""

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Receive any message and forward it
        in_message = op_input.receive("in")
        if in_message is not None:
            op_output.emit(in_message, "out")


class MultiPingRxOp(Operator):
    """Custom PingRxOp that can receive from multiple sources."""

    def __init__(self, fragment, *args, **kwargs):
        self.count = 1
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        # Define multi-receiver input port that can accept any number of connections
        spec.input("receivers", size=IOSpec.ANY_SIZE)

    def compute(self, op_input, op_output, context):
        value_vector = op_input.receive("receivers")
        if value_vector is not None:
            print(f"Rx message received (count: {self.count}, size: {len(value_vector)})")

            for i, value in enumerate(value_vector):
                print(f"Rx message value[{i}]: {value}")

            self.count += 1


class PingTxSubgraph(Subgraph):
    """Subgraph containing a single PingTxOp transmitter."""

    def __init__(self, fragment, instance_name):
        super().__init__(fragment, instance_name)

    def compose(self):
        # Create a PingTxOp with a count condition (send 8 messages total)
        tx_op = PingTxOp(
            self,
            CountCondition(self, count=8),
            name="transmitter",
        )

        forwarding_op = ForwardingOp(self, name="forwarding")

        # Add the operators to this subgraph
        self.add_flow(tx_op, forwarding_op, {("out", "in")})

        # Expose the "out" port so external operators can connect to it
        self.add_output_interface_port("data_out", forwarding_op, "out")


class MultiPingRxSubgraph(Subgraph):
    """Subgraph containing a multi-receiver MultiPingRxOp."""

    def __init__(self, fragment, instance_name):
        super().__init__(fragment, instance_name)

    def compose(self):
        # Create a multi-receiver MultiPingRxOp
        rx_op = MultiPingRxOp(self, name="multi_receiver")

        # Add the operator to this subgraph
        self.add_operator(rx_op)

        # Expose the "receivers" port so multiple external operators can connect to it
        self.add_input_interface_port("data_in", rx_op, "receivers")


class PingRxSubgraph(Subgraph):
    """Subgraph containing a single-receiver PingRxOp."""

    def __init__(self, fragment, instance_name):
        super().__init__(fragment, instance_name)

    def compose(self):
        # Create a single-receiver PingRxOp
        rx_op = PingRxOp(self, name="receiver")

        # Add the operator to this subgraph
        self.add_operator(rx_op)

        # Expose the "in" port so external operators can connect to it
        self.add_input_interface_port("data_in", rx_op, "in")


class MultiPingApplication(Application):
    """
    Application demonstrating Subgraph reusability with multiple instances.

    This application creates:
    - 3 instances of PingTxSubgraph (will create operators named "tx1_transmitter",
      "tx2_transmitter", "tx4_transmitter", etc.)
    - 1 instance of PingTxOp operator "tx2"
    - 1 instance of MultiPingRxSubgraph (creates operator named multi_rx_multi_receiver)
    - All transmitters connect to the single multi-receiver via exposed ports
    """

    def compose(self):
        # Create 3 transmitter subgraphs and one standalone transmitter operator
        tx_instance1 = PingTxSubgraph(self, "tx1")
        tx2 = PingTxOp(self, CountCondition(self, count=8), name="tx2")
        tx_instance3 = PingTxSubgraph(self, "tx3")
        tx_instance4 = PingTxSubgraph(self, "tx4")

        # Create one instance of the multi-receiver subgraph
        # This will create internal operator: multi_rx_multi_receiver
        rx_instance = MultiPingRxSubgraph(self, "multi_rx")

        # Subgraphs are automatically added to the Fragment when created

        # Connect all transmitters to the multi-receiver using exposed port names
        # Each PingTxSubgraph's "data_out" port (maps to internal "out" port) connects
        # to the receiver's "data_in" port (maps to internal "receivers" port)
        self.add_flow(tx_instance1, rx_instance, {("data_out", "data_in")})
        self.add_flow(tx2, rx_instance, {("out", "data_in")})
        self.add_flow(tx_instance3, rx_instance, {("data_out", "data_in")})
        self.add_flow(tx_instance4, rx_instance, {("data_out", "data_in")})

        print("Application composed with Subgraph instances:")


def main():
    app = MultiPingApplication()

    # optional code to visualize the flattened graph and port mapping
    app.compose_graph()
    port_map_yaml = app.graph.port_map_description()
    print("====== PORT MAPPING =======")
    print(port_map_yaml)

    # run the application
    app.run()


if __name__ == "__main__":
    main()
