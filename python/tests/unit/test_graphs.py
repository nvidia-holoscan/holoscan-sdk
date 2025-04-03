"""
SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from holoscan.core import Graph, Operator, OperatorGraph
from holoscan.graphs import FlowGraph, OperatorFlowGraph
from holoscan.operators import PingRxOp, PingTxOp


class ForwardOp(Operator):
    def setup(self, spec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("in")
        op_output.emit(msg, "out")


class TestOperatorGraph:
    def _get_tx_rx_ops(self, fragment):
        op_tx = PingTxOp(fragment, name="op_tx")
        op_rx = PingRxOp(fragment, name="op_rx")
        return op_tx, op_rx

    def test_name_alias(self):
        assert FlowGraph is OperatorFlowGraph

    def test_init(self):
        graph = OperatorFlowGraph()
        assert isinstance(graph, OperatorGraph)
        # Graph is also available as an alias for OperatorGraph for backwards compatibility
        assert isinstance(graph, Graph)

    def test_context(self):
        graph = OperatorFlowGraph()
        assert graph.context is None

    def test_add_node(self, fragment):
        graph = OperatorFlowGraph()
        op_tx, op_rx = self._get_tx_rx_ops(fragment)

        graph.add_node(op_tx)
        graph.add_node(op_rx)

        # Note: is_root and is_leaf will segfault if the input operator
        #       has not been added to the graph!
        assert graph.is_root(op_tx)
        assert graph.is_root(op_rx)
        assert graph.is_leaf(op_tx)
        assert graph.is_leaf(op_rx)

    def test_get_port_map(self, fragment):
        graph = OperatorFlowGraph()
        op_tx, op_rx = self._get_tx_rx_ops(fragment)

        fragment.add_flow(op_tx, op_rx)
        graph = fragment.graph

        port_map = graph.get_port_map(op_tx, op_rx)
        assert isinstance(port_map, dict)

    def test_get_root_operators(self, fragment):
        op_tx, op_rx = self._get_tx_rx_ops(fragment)

        # add_flow will automatically add the operators to the graph
        fragment.add_flow(op_tx, op_rx)

        # generate the graph
        graph = fragment.graph

        # test that the only root operator is the transmit operator
        root_ops = graph.get_root_nodes()
        assert len(root_ops) == 1
        root_op = root_ops[0]
        assert root_op is op_tx

        # add a second transmitter
        op_tx2 = PingTxOp(fragment, name="op_tx2")
        graph.add_node(op_tx2)

        root_ops = graph.get_root_nodes()
        assert len(root_ops) == 2
        assert set(root_ops) == {op_tx, op_tx2}

    def test_get_nodes(self, fragment):
        op_tx, op_rx = self._get_tx_rx_ops(fragment)

        # add_flow will automatically add the nodes to the graph
        fragment.add_flow(op_tx, op_rx)

        # generate the graph
        graph = fragment.graph

        # test that the only root operator is the transmit operator
        ops = graph.get_nodes()
        assert len(ops) == 2
        assert set(ops) == {op_tx, op_rx}

        # add a second pair of transmit/receive ops
        op_rx2 = PingRxOp(fragment, name="op_rx2")
        op_tx2 = PingTxOp(fragment, name="op_tx2")
        fragment.add_flow(op_tx2, op_rx2)

        ops = graph.get_nodes()
        assert len(ops) == 4
        assert set(ops) == {op_rx, op_tx, op_rx2, op_tx2}

    def test_get_next_nodes(self, fragment):
        op_tx, op_rx = self._get_tx_rx_ops(fragment)

        # add_flow will automatically add the nodes to the graph
        fragment.add_flow(op_tx, op_rx)

        # generate the graph
        graph = fragment.graph

        # test that the only root operator is the transmit operator
        next_ops = graph.get_next_nodes(op_tx)
        assert len(next_ops) == 1
        next_op = next_ops[0]
        assert next_op is op_rx

        # now add a second downstream operator to op_tx
        op_rx2 = PingRxOp(fragment, name="op_rx2")
        fragment.add_flow(op_tx, op_rx2)

        next_ops = graph.get_next_nodes(op_tx)
        assert len(next_ops) == 2
        assert set(next_ops) == {op_rx, op_rx2}

    def test_get_previous_nodes(self, fragment):
        op_tx, op_rx = self._get_tx_rx_ops(fragment)

        # add_flow will automatically add the operators to the graph
        fragment.add_flow(op_tx, op_rx)

        # generate the graph
        graph = fragment.graph

        # test that tx is the only previous node of rx
        prev_ops = graph.get_previous_nodes(op_rx)
        assert len(prev_ops) == 1
        prev_op = prev_ops[0]
        assert prev_op is op_tx

    def test_remove_node(self, fragment):
        op_tx, op_rx = self._get_tx_rx_ops(fragment)

        op_mx = ForwardOp(fragment, name="mx")

        # add_flow will automatically add the operators to the graph
        fragment.add_flow(op_tx, op_mx)
        fragment.add_flow(op_mx, op_rx)

        # generate the graph
        graph = fragment.graph
        assert len(graph.get_nodes()) == 3

        # test that mx is the only previous node of rx
        prev_ops = graph.get_previous_nodes(op_rx)
        assert len(prev_ops) == 1
        prev_op = prev_ops[0]
        assert prev_op is op_mx

        # test that mx is the only next node of tx
        next_ops = graph.get_next_nodes(op_tx)
        assert len(next_ops) == 1
        next_op = next_ops[0]
        assert next_op is op_mx

        # rx is a not a root node
        assert not graph.is_root(op_rx)

        graph.remove_node(op_mx)

        # the node was removed
        all_nodes = graph.get_nodes()
        assert op_mx not in all_nodes
        assert len(all_nodes) == 2

        # the node's edges were also removed
        assert len(graph.get_next_nodes(op_rx)) == 0
        assert len(graph.get_previous_nodes(op_tx)) == 0

        # rx is now a root node
        assert graph.is_root(op_rx)

        # connect tx directly to rx
        fragment.add_flow(op_tx, op_rx)

        # after the new add-flow call there is a new edge
        assert len(graph.get_next_nodes(op_tx)) == 1
        assert len(graph.get_previous_nodes(op_rx)) == 1

        # rx is once again not a root node
        assert not graph.is_root(op_rx)

    def test_dynamic_attribute_not_allowed(self):
        obj = OperatorFlowGraph()
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5
