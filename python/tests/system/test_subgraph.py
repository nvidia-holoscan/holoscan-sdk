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

import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, Fragment, IOSpec, Operator, OperatorSpec, Subgraph, Tracker
from holoscan.decorator import create_op
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
            # Convert to list if it's not already
            if not isinstance(value_vector, list):
                value_vector = [value_vector]

            print(f"Rx message received (count: {self.count}, size: {len(value_vector)})")

            for i, value in enumerate(value_vector):
                print(f"Rx message value[{i}]: {value}")

            self.count += 1


class PingTxSubgraph(Subgraph):
    """Subgraph containing a single PingTxOp transmitter."""

    def __init__(self, fragment, name):
        super().__init__(fragment, name)

    def compose(self):
        # Create a PingTxOp with a count condition
        tx_op = PingTxOp(
            self,
            CountCondition(self, count=8),  # Send 8 messages per transmitter
            name="transmitter",
        )

        forwarding_op = ForwardingOp(self, name="forwarding")

        # not necessary, but shouldn't hurt if add_operator was called explicitly before add_flow
        self.add_operator(tx_op)

        # Add the operators to this subgraph
        self.add_flow(tx_op, forwarding_op, {("out", "in")})

        # Expose the "out" port so external operators can connect to it
        self.add_output_interface_port("data_out", forwarding_op, "out")


class MultiPingRxSubgraph(Subgraph):
    """Subgraph containing a multi-receiver MultiPingRxOp."""

    def __init__(self, fragment, name):
        super().__init__(fragment, name)

    def compose(self):
        # Create a multi-receiver MultiPingRxOp
        rx_op = MultiPingRxOp(self, name="multi_receiver")

        # Add the operator to this subgraph
        self.add_operator(rx_op)

        # Expose the "receivers" port so multiple external operators can connect to it
        self.add_input_interface_port("data_in", rx_op, "receivers")


class PingRxSubgraph(Subgraph):
    """Subgraph containing a single-receiver PingRxOp."""

    def __init__(self, fragment, name):
        super().__init__(fragment, name)

    def compose(self):
        # Create a single-receiver PingRxOp
        rx_op = PingRxOp(self, name="receiver")

        # Add the operator to this subgraph
        self.add_operator(rx_op)

        # Expose the "in" port so external operators can connect to it
        self.add_input_interface_port("data_in", rx_op, "in")


class NestedTxSubgraph(Subgraph):
    """Nested Subgraph that contains PingTxSubgraph connected to ForwardingOp."""

    def __init__(self, fragment, name):
        super().__init__(fragment, name)

    def compose(self):
        # Create a nested PingTxSubgraph (auto-composes via Python __init__)
        ping_tx_subgraph = PingTxSubgraph(self, "ping_tx")

        # Create a ForwardingOp
        forwarding_op = ForwardingOp(self, name="forwarding")
        self.add_operator(forwarding_op)

        # Connect the nested subgraph to the forwarding operator
        self.add_flow(ping_tx_subgraph, forwarding_op, {("data_out", "in")})

        # Expose the forwarding operator's output as our interface
        self.add_output_interface_port("data_out", forwarding_op, "out")


class NestedRxSubgraph(Subgraph):
    """Nested Subgraph that contains ForwardingOp connected to PingRxSubgraph."""

    def __init__(self, fragment, name):
        super().__init__(fragment, name)

    def compose(self):
        # Create a ForwardingOp
        forwarding_op = ForwardingOp(self, name="forwarding")

        # not necessary, but shouldn't hurt if add_operator was called explicitly before add_flow
        self.add_operator(forwarding_op)

        # Create a nested PingRxSubgraph (auto-composes via Python __init__)
        ping_rx_subgraph = PingRxSubgraph(self, "ping_rx")

        # Connect the forwarding operator to the nested subgraph
        self.add_flow(forwarding_op, ping_rx_subgraph, {("out", "data_in")})

        # Expose the forwarding operator's input as our interface
        self.add_input_interface_port("data_in", forwarding_op, "in")


class DoubleNestedRxSubgraph(Subgraph):
    """
    Double nested Subgraph that contains only NestedRxSubgraph.

    This demonstrates exposing a nested subgraph's interface port directly as the parent's
    interface port using the add_input_interface_port overload that accepts Subgraph.
    """

    def __init__(self, fragment, name):
        super().__init__(fragment, name)

    def compose(self):
        # Create a nested NestedRxSubgraph (auto-composes via Python __init__)
        nested_rx_subgraph = NestedRxSubgraph(self, "nested_rx")

        # Expose the nested subgraph's interface port as our own interface port
        # This uses the overload that takes Subgraph
        self.add_input_interface_port("data_receiver", nested_rx_subgraph, "data_in")


class SubgraphPingApplication(Application):
    """
    Application demonstrating Subgraph reusability with multiple instances.

    This application creates:
    - 3 instances of PingTxSubgraph + 1 PingTxOp operator
    - 3 instances of PingRxSubgraph + 1 PingRxOp operator
    - Various 1:1 connection patterns between operators and subgraphs
    """

    def compose(self):
        # Create 3 transmitter subgraphs and one transmitter operator
        tx_instance1 = PingTxSubgraph(self, "tx1")
        tx2 = PingTxOp(self, CountCondition(self, count=8), name="tx2")
        tx_instance3 = PingTxSubgraph(self, "tx3")
        tx_instance4 = PingTxSubgraph(self, "tx4")

        # Create 3 receiver subgraphs and one receiver operator
        rx_instance1 = PingRxSubgraph(self, "rx1")
        rx_instance2 = PingRxSubgraph(self, "rx2")
        rx3 = PingRxOp(self, name="rx3")
        rx_instance4 = PingRxSubgraph(self, "rx4")

        # Make various types of 1:1 connections
        self.add_flow(tx_instance1, rx_instance1, {("data_out", "data_in")})
        self.add_flow(tx2, rx_instance2, {("out", "data_in")})
        self.add_flow(tx_instance3, rx3, {("data_out", "in")})
        self.add_flow(tx_instance4, rx_instance4, {("data_out", "data_in")})


class MultiPingApplication(Application):
    """
    Application demonstrating Subgraph reusability with multi-receiver pattern.

    This application creates:
    - 3 instances of PingTxSubgraph + 1 PingTxOp operator
    - 1 instance of MultiPingRxSubgraph
    - All transmitters connect to the single multi-receiver via exposed ports
    """

    def compose(self):
        # Create 3 instances of the transmitter subgraph and one transmitter operator
        tx_instance1 = PingTxSubgraph(self, "tx1")
        tx2 = PingTxOp(self, CountCondition(self, count=8), name="tx2")
        tx_instance3 = PingTxSubgraph(self, "tx3")
        tx_instance4 = PingTxSubgraph(self, "tx4")

        rx_instance = MultiPingRxSubgraph(self, "multi_rx")

        # Create N:1 connections from operators and subgraphs to the multi_rx subgraph
        self.add_flow(tx_instance1, rx_instance, {("data_out", "data_in")})
        self.add_flow(tx2, rx_instance, {("out", "data_in")})
        self.add_flow(tx_instance3, rx_instance, {("data_out", "data_in")})
        self.add_flow(tx_instance4, rx_instance, {("data_out", "data_in")})


class NestedSubgraphApplication(Application):
    """
    Application demonstrating nested Subgraphs.

    Architecture: NestedTxSubgraph -> ForwardingOp -> NestedRxSubgraph
    Where:
    - NestedTxSubgraph contains: PingTxSubgraph -> ForwardingOp
    - NestedRxSubgraph contains: ForwardingOp -> PingRxSubgraph
    """

    def compose(self):
        # Create nested subgraphs
        nested_tx = NestedTxSubgraph(self, "nested_tx")
        nested_rx = NestedRxSubgraph(self, "nested_rx")

        # Create a middle ForwardingOp to connect the nested subgraphs
        middle_forwarding = ForwardingOp(self, name="middle_forwarding")

        # Connect: NestedTxSubgraph -> ForwardingOp -> NestedRxSubgraph
        self.add_flow(nested_tx, middle_forwarding, {("data_out", "in")})
        self.add_flow(middle_forwarding, nested_rx, {("out", "data_in")})


class DoubleNestedSubgraphApplication(Application):
    """
    Minimal application demonstrating double-nested subgraph interface port exposure.

    This application creates:
    - 1 PingTxOp transmitter
    - 1 DoubleNestedRxSubgraph that exposes its nested subgraph's interface port

    Architecture:
    PingTxOp -> DoubleNestedRxSubgraph (contains NestedRxSubgraph (contains ForwardingOp ->
    PingRxSubgraph))
    """

    def compose(self):
        # Create a simple transmitter
        tx = PingTxOp(self, CountCondition(self, count=5), name="tx")

        # Create the double-nested receiver subgraph
        double_nested_rx = DoubleNestedRxSubgraph(self, "double_nested_rx")

        # Connect transmitter to double-nested receiver using exposed interface port
        self.add_flow(tx, double_nested_rx, {("out", "data_receiver")})


class TestSubgraphAddFlow:
    """Test class for Subgraph functionality."""

    def test_subgraph_connections(self):
        """Test connecting Subgraphs."""
        fragment = Fragment()

        tx_sg = PingTxSubgraph(fragment, "tx1")
        rx_sg = PingRxSubgraph(fragment, "rx1")

        # Connect Subgraph to Subgraph
        fragment.add_flow(tx_sg, rx_sg, {("data_out", "data_in")})

        # Verify the connection was made
        port_map = fragment.graph.port_map_description()
        assert "tx1_forwarding.out" in port_map
        assert "rx1_receiver.in" in port_map

    def test_mixed_operator_subgraph_connections(self):
        """Test connecting operators and Subgraphs."""
        fragment = Fragment()

        # Create operator and Subgraph
        tx_op = PingTxOp(fragment, CountCondition(fragment, count=8), name="tx_op")
        rx_sg = PingRxSubgraph(fragment, "rx1")

        # Connect Operator to Subgraph
        fragment.add_flow(tx_op, rx_sg, {("out", "data_in")})

        # Verify the connection was made
        port_map = fragment.graph.port_map_description()
        assert "tx_op.out" in port_map
        assert "rx1_receiver.in" in port_map

    def test_multi_receiver_pattern(self):
        """Test multi-receiver Subgraph pattern."""
        fragment = Fragment()

        # Create multiple transmitters and one multi-receiver
        tx1 = PingTxSubgraph(fragment, "tx1")
        tx2 = PingTxSubgraph(fragment, "tx2")
        multi_rx = MultiPingRxSubgraph(fragment, "multi_rx")

        # Connect all transmitters to the multi-receiver
        fragment.add_flow(tx1, multi_rx, {("data_out", "data_in")})
        fragment.add_flow(tx2, multi_rx, {("data_out", "data_in")})

        # Verify connections
        port_map = fragment.graph.port_map_description()
        assert "tx1_forwarding.out" in port_map
        assert "tx2_forwarding.out" in port_map
        assert "multi_rx_multi_receiver.receivers" in port_map


class TestSubgraphApplications:
    """Test class for Subgraph applications."""

    def test_subgraph_ping_application(self):
        """Test the SubgraphPingApplication (one-to-one pattern)."""
        app = SubgraphPingApplication()

        # Compose the application
        app.compose_graph()

        # Check that the expected operators were created
        nodes = app.graph.get_nodes()
        node_names = [node.name for node in nodes]

        # Should have qualified operator names
        expected_names = [
            "tx1_transmitter",
            "tx1_forwarding",
            "tx2",  # direct operator
            "tx3_transmitter",
            "tx3_forwarding",
            "tx4_transmitter",
            "tx4_forwarding",
            "rx1_receiver",
            "rx2_receiver",
            "rx3",  # direct operator
            "rx4_receiver",
        ]

        for name in expected_names:
            assert name in node_names, f"Expected operator '{name}' not found in {node_names}"

    def test_multi_ping_application(self):
        """Test the MultiPingApplication (multi-to-one pattern)."""
        app = MultiPingApplication()

        # Compose the application
        app.compose_graph()

        # Check that the expected operators were created
        nodes = app.graph.get_nodes()
        node_names = [node.name for node in nodes]

        # Should have qualified operator names
        expected_names = [
            "tx1_transmitter",
            "tx1_forwarding",
            "tx2",  # direct operator
            "tx3_transmitter",
            "tx3_forwarding",
            "tx4_transmitter",
            "tx4_forwarding",
            "multi_rx_multi_receiver",
        ]

        for name in expected_names:
            assert name in node_names, f"Expected operator '{name}' not found in {node_names}"

    def test_nested_subgraph_application(self):
        """Test the NestedSubgraphApplication (nested pattern)."""
        app = NestedSubgraphApplication()

        # Compose the application
        app.compose_graph()

        # Check that the expected operators were created
        nodes = app.graph.get_nodes()
        node_names = [node.name for node in nodes]

        # Should have hierarchical qualified operator names
        expected_names = [
            # From nested_tx (NestedTxSubgraph)
            "nested_tx_ping_tx_transmitter",  # PingTxSubgraph -> PingTxOp
            "nested_tx_ping_tx_forwarding",  # PingTxSubgraph -> ForwardingOp
            "nested_tx_forwarding",  # NestedTxSubgraph -> ForwardingOp
            # Middle operator
            "middle_forwarding",
            # From nested_rx (NestedRxSubgraph)
            "nested_rx_forwarding",  # NestedRxSubgraph -> ForwardingOp
            "nested_rx_ping_rx_receiver",  # PingRxSubgraph -> PingRxOp
        ]

        for name in expected_names:
            assert name in node_names, f"Expected operator '{name}' not found in {node_names}"


@pytest.mark.parametrize("data_flow_tracking_enabled", [True, False])
def test_subgraph_ping_application_run(data_flow_tracking_enabled, capfd):
    """Test running the SubgraphPingApplication."""
    app = SubgraphPingApplication()

    if data_flow_tracking_enabled:
        with Tracker(app) as tracker:
            app.run()
            tracker.print()
    else:
        app.run()

    # Verify no errors during composition
    captured = capfd.readouterr()
    assert "error" not in captured.err.lower()
    # flow tracking result should be shown when track is True
    num_tracking_results = 1 if data_flow_tracking_enabled else 0
    assert captured.out.count("Data Flow Tracking Results:") == num_tracking_results
    if data_flow_tracking_enabled:
        # check that all expected paths are present in the tracking output
        assert "Total paths: 4" in captured.out
        assert "tx1_transmitter,tx1_forwarding,rx1_receiver" in captured.out
        assert "tx2,rx2_receiver" in captured.out
        assert "tx3_transmitter,tx3_forwarding,rx3" in captured.out
        assert "tx4_transmitter,tx4_forwarding,rx4_receiver" in captured.out


@pytest.mark.parametrize("data_flow_tracking_enabled", [True, False])
def test_multi_ping_application_run(data_flow_tracking_enabled, capfd):
    """Test running the MultiPingApplication."""
    app = MultiPingApplication()

    if data_flow_tracking_enabled:
        with Tracker(app) as tracker:
            app.run()
            tracker.print()
    else:
        app.run()

    # Verify no errors during composition
    captured = capfd.readouterr()
    assert "error" not in captured.err.lower()
    # flow tracking result should be shown when track is True
    num_tracking_results = 1 if data_flow_tracking_enabled else 0
    assert captured.out.count("Data Flow Tracking Results:") == num_tracking_results
    if data_flow_tracking_enabled:
        # check that all expected paths are present in the tracking output
        assert "Total paths: 4" in captured.out
        assert "tx1_transmitter,tx1_forwarding,multi_rx_multi_receiver" in captured.out
        assert "tx2,multi_rx_multi_receiver" in captured.out
        assert "tx3_transmitter,tx3_forwarding,multi_rx_multi_receiver" in captured.out
        assert "tx4_transmitter,tx4_forwarding,multi_rx_multi_receiver" in captured.out


@pytest.mark.parametrize("data_flow_tracking_enabled", [True, False])
def test_nested_subgraph_application_run(data_flow_tracking_enabled, capfd):
    """Test running the NestedSubgraphApplication."""
    app = NestedSubgraphApplication()

    if data_flow_tracking_enabled:
        with Tracker(app) as tracker:
            app.run()
            tracker.print()
    else:
        app.run()

    # Verify no errors during composition
    captured = capfd.readouterr()
    assert "error" not in captured.err.lower()
    # flow tracking result should be shown when track is True
    num_tracking_results = 1 if data_flow_tracking_enabled else 0
    assert captured.out.count("Data Flow Tracking Results:") == num_tracking_results
    if data_flow_tracking_enabled:
        assert "Total paths: 1" in captured.out
        assert (
            "Path 1: nested_tx_ping_tx_transmitter,nested_tx_ping_tx_forwarding,"
            "nested_tx_forwarding,middle_forwarding,nested_rx_forwarding,"
            "nested_rx_ping_rx_receiver"
        ) in captured.out


@pytest.mark.parametrize("data_flow_tracking_enabled", [True, False])
def test_double_nested_subgraph_application_run(data_flow_tracking_enabled, capfd):
    """Test running the DoubleNestedSubgraphApplication."""
    app = DoubleNestedSubgraphApplication()

    if data_flow_tracking_enabled:
        with Tracker(app) as tracker:
            app.run()
            tracker.print()
    else:
        app.run()

    # Verify no errors during composition
    captured = capfd.readouterr()
    assert "error" not in captured.err.lower()

    # Get the graph nodes to verify hierarchical naming
    node_names = [node.name for node in app.graph.get_nodes()]

    # The hierarchical naming should produce:
    # - tx (the transmitter)
    # - double_nested_rx_nested_rx_forwarding (ForwardingOp in NestedRxSubgraph)
    # - double_nested_rx_nested_rx_ping_rx_receiver (PingRxOp in PingRxSubgraph)
    expected_names = [
        "tx",
        "double_nested_rx_nested_rx_forwarding",
        "double_nested_rx_nested_rx_ping_rx_receiver",
    ]

    # Check that both lists contain exactly the same names
    assert sorted(node_names) == sorted(expected_names), (
        f"Node names don't match expected names.\n"
        f"Actual nodes: {sorted(node_names)}\n"
        f"Expected nodes: {sorted(expected_names)}\n"
    )

    # Check that messages were received through the double-nested subgraph chain
    # The message should pass through: PingTxOp -> NestedRxSubgraph.ForwardingOp ->
    # PingRxSubgraph.PingRxOp
    assert "Rx message value: 5" in captured.out, (
        f"Expected to find 'Rx message value: 5' in stderr output\n"
        f"=== STDOUT ===\n{captured.out}\n===========\n"
    )

    # flow tracking result should be shown when track is True
    num_tracking_results = 1 if data_flow_tracking_enabled else 0
    assert captured.out.count("Data Flow Tracking Results:") == num_tracking_results
    if data_flow_tracking_enabled:
        assert "Total paths: 1" in captured.out
        assert (
            "Path 1: tx,double_nested_rx_nested_rx_forwarding,"
            "double_nested_rx_nested_rx_ping_rx_receiver"
        ) in captured.out


# ========== Control Flow Tests with Subgraphs ==========


class SimpleExecOp(Operator):
    """Simple operator for control flow testing."""

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        pass

    def compute(self, op_input, op_output, context):
        print(f"SimpleExecOp: {self.name} executed")


class SequentialExecSubgraph(Subgraph):
    """Subgraph with sequential control flow and exposed execution interface ports."""

    def __init__(self, fragment, name):
        super().__init__(fragment, name)

    def compose(self):
        node2 = SimpleExecOp(self, name="node2")
        node3 = SimpleExecOp(self, name="node3")

        # Sequential execution control flow
        self.add_flow(node2, node3)

        # Expose execution control ports
        self.add_input_exec_interface_port("exec_in", node2)
        self.add_output_exec_interface_port("exec_out", node3)


class InputExecSubgraph(Subgraph):
    """Subgraph with only input execution interface port."""

    def __init__(self, fragment, name):
        super().__init__(fragment, name)

    def compose(self):
        node = SimpleExecOp(self, name="node")
        self.add_input_exec_interface_port("exec_in", node)


class OutputExecSubgraph(Subgraph):
    """Subgraph with both input and output execution interface ports (passthrough)."""

    def __init__(self, fragment, name):
        super().__init__(fragment, name)

    def compose(self):
        node = SimpleExecOp(self, name="node")
        # Need both input and output ports for the operator to be triggered and to trigger
        # downstream
        self.add_input_exec_interface_port("exec_in", node)
        self.add_output_exec_interface_port("exec_out", node)


class NestedExecSubgraph(Subgraph):
    """Nested Subgraph with execution interface ports."""

    def __init__(self, fragment, name):
        super().__init__(fragment, name)

    def compose(self):
        # Create a nested sequential subgraph
        sequential_sg = SequentialExecSubgraph(self, "sequential_sg")

        # Expose the nested subgraph's execution interface ports
        self.add_input_exec_interface_port("exec_in", sequential_sg, "exec_in")
        self.add_output_exec_interface_port("exec_out", sequential_sg, "exec_out")


class SequentialExecWithSubgraphApp(Application):
    """Application testing sequential execution flow with Subgraph."""

    def compose(self):
        node1 = SimpleExecOp(self, name="node1")
        sequential_sg = SequentialExecSubgraph(self, "sequential_sg")
        node4 = SimpleExecOp(self, name="node4")

        # Auto-resolves control flow connections
        self.add_flow(self.start_op(), node1)
        self.add_flow(node1, sequential_sg)
        self.add_flow(sequential_sg, node4)


class InputExecSubgraphApp(Application):
    """Application testing input execution interface port."""

    def compose(self):
        node1 = SimpleExecOp(self, name="node1")
        input_sg = InputExecSubgraph(self, "input_sg")

        self.add_flow(self.start_op(), node1)
        self.add_flow(node1, input_sg)  # Auto-resolves to exec_in


class OutputExecSubgraphApp(Application):
    """Application testing output execution interface port."""

    def compose(self):
        output_sg = OutputExecSubgraph(self, "output_sg")
        node2 = SimpleExecOp(self, name="node2")

        self.add_flow(self.start_op(), output_sg)
        self.add_flow(output_sg, node2)  # Auto-resolves from exec_out


class NestedExecSubgraphApp(Application):
    """Application testing nested execution Subgraphs."""

    def compose(self):
        node1 = SimpleExecOp(self, name="node1")
        nested_sg = NestedExecSubgraph(self, "nested_sg")
        node4 = SimpleExecOp(self, name="node4")

        self.add_flow(self.start_op(), node1)
        self.add_flow(node1, nested_sg)
        self.add_flow(nested_sg, node4)


def test_sequential_exec_with_subgraph(capfd):
    """Test sequential execution flow with Subgraph."""
    app = SequentialExecWithSubgraphApp()
    app.run()

    captured = capfd.readouterr()

    # Check that nodes are present
    node_names = {node.name for node in app.graph.get_nodes()}
    expected_names = {"<|start|>", "node1", "sequential_sg_node2", "sequential_sg_node3", "node4"}

    assert node_names == expected_names, (
        f"Node names don't match expected names.\n"
        f"Actual nodes: {node_names}\n"
        f"Expected nodes: {expected_names}\n"
    )

    # Check that all operators executed
    assert "SimpleExecOp: node1 executed" in captured.out
    assert "SimpleExecOp: sequential_sg_node2 executed" in captured.out
    assert "SimpleExecOp: sequential_sg_node3 executed" in captured.out
    assert "SimpleExecOp: node4 executed" in captured.out


def test_input_exec_interface_port(capfd):
    """Test input execution interface port."""
    app = InputExecSubgraphApp()
    app.run()

    captured = capfd.readouterr()

    # Check that nodes are present
    node_names = {node.name for node in app.graph.get_nodes()}
    expected_names = {"<|start|>", "input_sg_node", "node1"}

    assert node_names == expected_names, (
        f"Node names don't match expected names.\n"
        f"Actual nodes: {node_names}\n"
        f"Expected nodes: {expected_names}\n"
    )

    # Check that both operators executed
    assert "SimpleExecOp: input_sg_node executed" in captured.out
    assert "SimpleExecOp: node1 executed" in captured.out


def test_output_exec_interface_port(capfd):
    """Test output execution interface port."""
    app = OutputExecSubgraphApp()
    app.run()

    captured = capfd.readouterr()

    # Check that nodes are present
    node_names = {node.name for node in app.graph.get_nodes()}
    expected_names = {"<|start|>", "output_sg_node", "node2"}

    assert node_names == expected_names, (
        f"Node names don't match expected names.\n"
        f"Actual nodes: {node_names}\n"
        f"Expected nodes: {expected_names}\n"
    )

    # Check that both operators executed
    assert "SimpleExecOp: output_sg_node executed" in captured.out
    assert "SimpleExecOp: node2 executed" in captured.out


def test_nested_exec_subgraph(capfd):
    """Test nested execution Subgraphs."""
    app = NestedExecSubgraphApp()
    app.run()

    captured = capfd.readouterr()

    # Check that nodes are present
    node_names = {node.name for node in app.graph.get_nodes()}
    expected_names = {
        "<|start|>",
        "node1",
        "nested_sg_sequential_sg_node2",
        "nested_sg_sequential_sg_node3",
        "node4",
    }

    assert node_names == expected_names, (
        f"Node names don't match expected names.\n"
        f"Actual nodes: {node_names}\n"
        f"Expected nodes: {expected_names}\n"
    )

    # Check that all operators executed
    assert "SimpleExecOp: node1 executed" in captured.out
    assert "SimpleExecOp: nested_sg_sequential_sg_node2 executed" in captured.out
    assert "SimpleExecOp: nested_sg_sequential_sg_node3 executed" in captured.out
    assert "SimpleExecOp: node4 executed" in captured.out


# ========== Error/Edge Case Tests ==========


class DataPortOp(Operator):
    """Operator with explicit data ports (not exec ports)."""

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("data_in")
        spec.output("data_out")

    def compute(self, op_input, op_output, context):
        pass


class MultiExecInputSubgraph(Subgraph):
    """Subgraph with multiple input exec interface ports (should cause auto-connect error)."""

    def __init__(self, fragment, name):
        super().__init__(fragment, name)

    def compose(self):
        node1 = SimpleExecOp(self, name="node1")
        node2 = SimpleExecOp(self, name="node2")

        # Expose multiple input exec ports
        self.add_input_exec_interface_port("exec_in1", node1)
        self.add_input_exec_interface_port("exec_in2", node2)


class MultiExecOutputSubgraph(Subgraph):
    """Subgraph with multiple output exec interface ports (should cause auto-connect error)."""

    def __init__(self, fragment, name):
        super().__init__(fragment, name)

    def compose(self):
        node1 = SimpleExecOp(self, name="node1")
        node2 = SimpleExecOp(self, name="node2")

        # Add an input exec port so start_op can connect to it
        self.add_input_exec_interface_port("exec_in", node1)

        # Expose multiple output exec ports
        self.add_output_exec_interface_port("exec_out1", node1)
        self.add_output_exec_interface_port("exec_out2", node2)


def test_multiple_exec_input_ports_error(capfd):
    """Test that auto-connecting to subgraph with multiple exec input ports fails."""
    app = Application()

    def compose():
        node1 = SimpleExecOp(app, name="node1")
        multi_input_sg = MultiExecInputSubgraph(app, "multi_input_sg")

        app.add_flow(app.start_op(), node1)
        # This should fail because subgraph has multiple exec input ports
        app.add_flow(node1, multi_input_sg)

    app.compose = compose

    with pytest.raises(RuntimeError, match="more than one execution input port"):
        app.run()

    captured = capfd.readouterr()
    # Should also see error in stderr
    assert "more than one execution input port" in captured.err.lower()


def test_multiple_exec_output_ports_error(capfd):
    """Test that auto-connecting from subgraph with multiple exec output ports fails."""
    app = Application()

    def compose():
        multi_output_sg = MultiExecOutputSubgraph(app, "multi_output_sg")
        node2 = SimpleExecOp(app, name="node2")

        app.add_flow(app.start_op(), multi_output_sg)
        # This should fail because subgraph has multiple exec output ports
        app.add_flow(multi_output_sg, node2)

    app.compose = compose

    with pytest.raises(RuntimeError, match="more than one execution output port"):
        app.run()

    captured = capfd.readouterr()
    # Should also see error in stderr
    assert "more than one execution output port" in captured.err.lower()


def test_exec_port_name_conflict_error(capfd):
    """Test that duplicate exec interface port names are detected."""
    app = Application()

    def compose():
        # Create a subgraph with duplicate port name
        class DuplicatePortSubgraph(Subgraph):
            def __init__(self, fragment, name):
                super().__init__(fragment, name)

            def compose(self):
                node1 = SimpleExecOp(self, name="node1")
                node2 = SimpleExecOp(self, name="node2")

                # Add same port name twice - should fail
                self.add_input_exec_interface_port("exec_in", node1)
                self.add_input_exec_interface_port("exec_in", node2)  # Duplicate!

        DuplicatePortSubgraph(app, "dup_sg")

    app.compose = compose

    with pytest.raises(RuntimeError, match="already exists"):
        app.run()

    captured = capfd.readouterr()
    # Should also see error in stderr
    assert "already exists" in captured.err.lower()


def test_data_exec_port_name_conflict_error(capfd):
    """Test that exec port name conflicts with data port name are detected."""
    app = Application()

    def compose():
        # Create a subgraph where exec port name conflicts with data port
        class ConflictingPortSubgraph(Subgraph):
            def __init__(self, fragment, name):
                super().__init__(fragment, name)

            def compose(self):
                data_op = DataPortOp(self, name="data_op")
                exec_op = SimpleExecOp(self, name="exec_op")

                # Add data port first
                self.add_input_interface_port("port1", data_op, "data_in")
                # Try to add exec port with same name - should fail
                self.add_input_exec_interface_port("port1", exec_op)

        ConflictingPortSubgraph(app, "conflict_sg")

    app.compose = compose

    with pytest.raises(RuntimeError, match="already exists"):
        app.run()

    captured = capfd.readouterr()
    # Should also see error in stderr
    assert "already exists" in captured.err.lower()


def test_nested_exec_port_wrong_type_error(capfd):
    """Test that using data port name when exec port expected fails."""
    app = Application()

    def compose():
        # Create a parent subgraph trying to expose nested subgraph's data port as exec port
        class WrongTypeSubgraph(Subgraph):
            def __init__(self, fragment, name):
                super().__init__(fragment, name)

            def compose(self):
                # Create nested subgraph with data port
                class NestedDataSubgraph(Subgraph):
                    def __init__(self, fragment, name):
                        super().__init__(fragment, name)

                    def compose(self):
                        data_op = DataPortOp(self, name="data_op")
                        self.add_input_interface_port("data_in", data_op, "data_in")

                nested = NestedDataSubgraph(self, "nested")
                # Try to expose data port as exec port - should fail
                self.add_input_exec_interface_port("exec_in", nested, "data_in")

        WrongTypeSubgraph(app, "wrong_sg")

    app.compose = compose

    with pytest.raises(RuntimeError, match="data interface port"):
        app.run()

    captured = capfd.readouterr()
    # Should also see error in stderr
    assert "data interface port" in captured.err.lower()


def test_nonexistent_interface_port_error(capfd):
    """Test that referencing non-existent interface port fails."""
    app = Application()

    def compose():
        node1 = SimpleExecOp(app, name="node1")
        sequential_sg = SequentialExecSubgraph(app, "sequential_sg")

        app.add_flow(app.start_op(), node1)
        # Try to connect to non-existent port - should fail
        app.add_flow(node1, sequential_sg, {("__output_exec__", "nonexistent_port")})

    app.compose = compose

    with pytest.raises(RuntimeError, match="not found"):
        app.run()

    captured = capfd.readouterr()
    # Should also see error in stderr
    assert "not found" in captured.err.lower()


def test_auto_connect_no_ports_error(capfd):
    """Test that auto-connect fails when no compatible ports exist."""
    app = Application()

    def compose():
        # Create empty subgraph with no interface ports
        class EmptySubgraph(Subgraph):
            def __init__(self, fragment, name):
                super().__init__(fragment, name)

            def compose(self):
                # Create operator but don't expose any ports
                SimpleExecOp(self, name="node")

        node1 = SimpleExecOp(app, name="node1")
        empty_sg = EmptySubgraph(app, "empty_sg")

        app.add_flow(app.start_op(), node1)
        # Try to auto-connect to empty subgraph - should fail
        app.add_flow(node1, empty_sg)

    app.compose = compose

    with pytest.raises(RuntimeError, match="no data or execution interface ports"):
        app.run()

    captured = capfd.readouterr()
    # Should also see error in stderr
    assert "no data or execution interface ports" in captured.err.lower()


@create_op(outputs=["out_ping"])
def ping_decorator_op() -> str:
    return "ping"


class SubgraphWithDecoratorOp(Subgraph):
    # Subgraph where one of the operators is created via the decorator API

    def __init__(self, fragment, name):
        super().__init__(fragment, name)

    def compose(self):
        # Create a PingTxOp with a count condition
        tx_op = ping_decorator_op(self, CountCondition(self, count=8), name="tx")
        # rx = PingRxOp(self, name="rx")
        # self.add_flow(tx_op, rx)

        # Expose the "out_ping" port so external operators can connect to it
        self.add_output_interface_port("data_out", tx_op, "out_ping")


class SubgraphWithDecoratorOpApplication(Application):
    """
    Application demonstrating Subgraph with decorator API operator.

    This application creates:
    - SubgraphWithDecoratorOp containing a decorator API operator that transmits a string
    - PingRxSubgraph that receives the string
    """

    def compose(self):
        # Create 3 transmitter subgraphs and one transmitter operator
        tx_subgraph = SubgraphWithDecoratorOp(self, name="tx_sub")
        rx_subgraph = PingRxSubgraph(self, name="rx_sub")
        self.add_flow(tx_subgraph, rx_subgraph)


def test_subgraph_with_decorator_api(capfd):
    """Test that decorator API operators work correctly within a subgraph."""
    app = SubgraphWithDecoratorOpApplication()
    app.run()

    captured = capfd.readouterr()

    # Check that nodes are present
    node_names = {node.name for node in app.graph.get_nodes()}
    expected_names = {"tx_sub_tx", "rx_sub_receiver"}

    assert node_names == expected_names, (
        f"Node names don't match expected names.\n"
        f"Actual nodes: {node_names}\n"
        f"Expected nodes: {expected_names}\n"
    )
    assert captured.out.count("Rx message value: ping") == 8
