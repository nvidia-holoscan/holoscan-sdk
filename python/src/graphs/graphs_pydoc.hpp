/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PYHOLOSCAN_GRAPHS_PYDOC_HPP
#define PYHOLOSCAN_GRAPHS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace Graph {

// Constructor
PYDOC(Graph, R"doc(
Abstract base class for all graphs
)doc")

}  // namespace Graph

namespace FlowGraph {

// Constructor
PYDOC(FlowGraph, R"doc(
Directed acyclic graph (DAG) class.
)doc")

PYDOC(add_operator, R"doc(
Add a node to the graph.

Parameters
----------
op : holoscan.core.Operator
    The node to add.
)doc")

PYDOC(add_flow, R"doc(
Add an edge to the graph.

Parameters
----------
op_u : Operator
    The source operator.
op_v : Operator
    The destination operator.
port_map : dict
    A map from the source operator's port name to the destination operator's port
    name(s).
)doc")

PYDOC(get_port_map, R"doc(
Get a port mapping dictionary between two operators in the graph.

Parameters
----------
op_u : holoscan.core.Operator
    The source operator.
op_v : holoscan.core.Operator
    The destination operator.

Returns
-------
port_map : dict
    A map from the source operator's port name to the destination operator's port
    name(s).
)doc")

PYDOC(is_root, R"doc(
Check if an operator is a root operator.

Parameters
----------
op : holoscan.core.Operator
    A node in the graph.

Returns
-------
bool
    Whether the operator is a root operator
)doc")

PYDOC(is_leaf, R"doc(
Check if an operator is a leaf operator.

Parameters
----------
op : holoscan.core.Operator
    A node in the graph.

Returns
-------
bool
    Whether the operator is a leaf operator
)doc")

PYDOC(get_root_operators, R"doc(
Get all root operators.

Returns
-------
list of Operator
    A list containing all root operators.
)doc")

PYDOC(get_operators, R"doc(
Get all operators.

Returns
-------
list of Operator
    A list containing all operators.
)doc")

PYDOC(get_next_operators, R"doc(
Get the operators immediately downstream of a given operator.

Parameters
----------
op : holoscan.core.Operator
    A node in the graph.

Returns
-------
list of Operator
    A list containing the downstream operators.
)doc")

PYDOC(context, R"doc(
The graph's context (as an opaque PyCapsule object)
)doc")

}  // namespace FlowGraph

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_GRAPHS_PYDOC_HPP
