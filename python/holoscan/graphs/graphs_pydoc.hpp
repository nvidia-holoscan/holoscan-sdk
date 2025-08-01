/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Directed graph class.
)doc")

PYDOC(add_node, R"doc(
Add the node to the graph.

Parameters
----------
node : holoscan.core.Operator | holoscan.core.Fragment
    The node to add.
)doc")

PYDOC(remove_node, R"doc(
Remove the node to the graph.

Parameters
----------
node : holoscan.core.Operator | holoscan.core.Fragment
    The node to remove. This will also remove all edges connected to the node.
)doc")

PYDOC(get_port_map, R"doc(
Get a port mapping dictionary between two nodes in the graph.

Parameters
----------
node_u : holoscan.core.Operator | holoscan.core.Fragment
    The source node.
node_v : holoscan.core.Operator | holoscan.core.Fragment
    The destination node.

Returns
-------
port_map : dict
    A map from the source node's port name to the destination node's port
    name(s). The keys of the dictionary are strings and the values are sets of
    strings.
)doc")

PYDOC(is_root, R"doc(
Check if the node is a root node.

Parameters
----------
node : holoscan.core.Operator | holoscan.core.Fragment
    A node in the graph.

Returns
-------
bool
    Whether the node is a root node
)doc")

PYDOC(is_user_defined_root, R"doc(
Check if the node is a user-defined root node.

A user-defined root is the first node added to the graph.

Parameters
----------
node : holoscan.core.Operator | holoscan.core.Fragment
    A node in the graph.

Returns
-------
bool
    Whether the node is a user-defined root node
)doc")

PYDOC(is_leaf, R"doc(
Check if the node is a leaf node.

Parameters
----------
node : holoscan.core.Operator | holoscan.core.Fragment
    A node in the graph.

Returns
-------
bool
    Whether the node is a leaf node
)doc")

PYDOC(has_cycle, R"doc(
Get all root nodes of the cycles if the graph has cycle(s). Otherwise, an empty list is returned.

Returns
-------
list of Operator or Fragment
    A list containing all root nodes of the cycles in the graph.
)doc")

PYDOC(get_root_nodes, R"doc(
Get all root nodes.

Returns
-------
list of Operator or Fragment
    A list containing all root nodes.
)doc")

PYDOC(get_nodes, R"doc(
Get all nodes.

The nodes are returned in the order they were added to the graph.

Returns
-------
list of Operator or Fragment
    A list containing all nodes.
)doc")

PYDOC(get_next_nodes, R"doc(
Get the nodes immediately downstream of a given node.

Parameters
----------
node : holoscan.core.Operator | holoscan.core.Fragment
    A node in the graph.

Returns
-------
list of Operator or Fragment
    A list containing the downstream nodes.
)doc")

PYDOC(get_previous_nodes, R"doc(
Get the nodes immediately upstream of a given node.

Parameters
----------
node : holoscan.core.Operator | holoscan.core.Fragment
    A node in the graph.

Returns
-------
list of Operator or Fragment
    A list containing the upstream nodes.
)doc")

PYDOC(context, R"doc(
The graph's context (as an opaque PyCapsule object)
)doc")

PYDOC(get_port_connectivity_maps, R"doc(
Get port connectivity maps for the graph.

Returns a tuple of two dictionaries:
1. Input ports to connected output ports mapping
2. Output ports to connected input ports mapping

Each port is identified by a unique string in the format "fragment.operator.port"
(or "fragment.port" for FragmentFlowGraph).

Returns
-------
tuple[dict[str, list[str]], dict[str, list[str]]]
    A tuple containing (input_to_output_map, output_to_input_map) where:
    - input_to_output_map: Maps input port IDs to lists of connected output port IDs
    - output_to_input_map: Maps output port IDs to lists of connected input port IDs
)doc")

PYDOC(port_map_description, R"doc(
Get a YAML-formatted description of the port connectivity maps.

Returns a human-readable YAML string that describes the port connectivity
of the graph.

Returns
-------
str
    A YAML-formatted string describing the port connectivity maps
)doc")

}  // namespace FlowGraph

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_GRAPHS_PYDOC_HPP */
