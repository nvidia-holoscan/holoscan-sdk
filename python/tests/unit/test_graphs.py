# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from test_core import get_tx_and_rx_ops

from holoscan.core import Fragment, Graph
from holoscan.graphs import FlowGraph


class TestGraph:
    def test_init(self):
        graph = FlowGraph()
        assert isinstance(graph, Graph)

    def test_context(self):
        graph = FlowGraph()
        assert graph.context is None

    def test_add_operator(self):
        graph = FlowGraph()
        fragment = Fragment()
        op_tx, op_rx = get_tx_and_rx_ops(fragment)

        graph.add_operator(op_tx)
        graph.add_operator(op_rx)

        # Note: is_root and is_leaf will segfault if the input operator
        #       has not been added to the graph!
        assert graph.is_root(op_tx)
        assert graph.is_root(op_rx)
        assert graph.is_leaf(op_tx)
        assert graph.is_leaf(op_rx)

    def test_dynamic_attribute_not_allowed(self):
        obj = FlowGraph()
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5

    # TODO: add_flow
    #    Due to use of PYBIND11_MAKE_OPAQUE in the bindings, can't pass
    #    arguments of type EdgeDataType. These need STL containers, but we
    #    apparently can't include pybind11/stl.h while also using
    #    PYBIND11_MAKE_OPAQUE?
    # def test_add_flow(self):
    #     graph = FlowGraph()
    #     fragment = Fragment()
    #     op_tx, op_rx = get_tx_and_rx_ops(fragment)
    #     graph.add_operator(op_tx)
    #     graph.add_operator(op_rx)
    #     graph.add_flow(op_tx, op_rx, {"tensor": ("in2",)})

    # TODO: test other methods?
