/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CORE_SUBGRAPH_PYDOC_HPP
#define PYHOLOSCAN_CORE_SUBGRAPH_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace Subgraph {

// Constructor
PYDOC(Subgraph, R"doc(
A reusable subgraph that directly populates a Fragment's operator graph.

Subgraph receives Fragment* during construction and directly adds operators and flows to the
Fragment's main graph during compose().

Parameters
----------
subgraph : object
    The Python subgraph object (self)
fragment : Fragment
    Target Fragment to populate with operators
name : str
    Unique instance name for operator qualification

Examples
--------
.. code-block:: python

    class CameraSubgraph(Subgraph):
        def __init__(self, fragment, name):
            super().__init__(fragment, name)

        def compose(self):
            source = PingTxOp(self, name="source")
            converter = FormatConverterOp(self, name="converter")

            self.add_flow(source, converter)
            self.add_output_interface_port("video_out", converter, "tensor")

    # In Fragment.compose():
    camera1 = CameraSubgraph(self, "camera1")
    camera2 = CameraSubgraph(self, "camera2")
    visualizer = HolovizOp(self, name="visualizer")

    self.add_flow(camera1, visualizer, [("video_out", "receivers")])
    self.add_flow(camera2, visualizer, [("video_out", "receivers")])
)doc")

// Properties
PYDOC(name, R"doc(
    Get the name for this subgraph.

    Returns
    -------
    str
        The name used for operator qualification
    )doc")

PYDOC(fragment, R"doc(
Get the Fragment that this subgraph belongs to.

Returns
-------
Fragment
    The Fragment that contains this subgraph
)doc")

PYDOC(is_composed, R"doc(
Check if the subgraph has been composed.

Returns
-------
bool
    True if the subgraph has been composed, False otherwise
)doc")

PYDOC(set_composed, R"doc(
Set the composed state of the subgraph.

Parameters
----------
composed : bool
    The composed state to set
)doc")

// Methods
PYDOC(compose, R"doc(
Define the internal structure of the subgraph.

This method should create operators and flows, which will be directly added to the Fragment's
main graph with qualified names.
)doc")

PYDOC(add_operator, R"doc(
Add an operator to the Fragment's main graph with qualified name.

This directly calls fragment.add_operator() with a qualified name,
eliminating the need for intermediate graph storage.

Parameters
----------
op : Operator
    The operator to add
)doc")

// add_flow method (single docstring for all overloads)
PYDOC(add_flow, R"doc(
Connect components within this Subgraph.

Parameters
----------
upstream_subgraph : holoscan.core.Operator or holoscan.core.Subgraph
    The upstream subgraph
downstream_subgraph : holoscan.core.Operator or holoscan.core.Subgraph
    The downstream subgraph
port_pairs : list of tuple of str, optional
    Port connections as (upstream_interface_port, downstream_interface_port) pairs
connector_type : holoscan.core.IOSpec.ConnectorType, optional
    The connector type to use for the connection

Notes
-----
This is an overloaded function. Additional variants exist:

1.) Operator to Operator: Connect two operators directly
    - add_flow(upstream_op, downstream_op, port_pairs=None)

2.) Operator to Subgraph: Connect an operator to a subgraph
    - add_flow(upstream_op, downstream_subgraph, port_pairs=None)

3.) Subgraph to Operator: Connect a subgraph to an operator
    - add_flow(upstream_subgraph, downstream_op, port_pairs=None)

4.) Subgraph to Subgraph: Connect a subgraph to a subgraph
    - add_flow(upstream_subgraph, downstream_subgraph, port_pairs=None)

5.) All variants also support connector_type parameter:
    - add_flow(..., connector_type)
    - add_flow(..., port_pairs, connector_type)

6.) When port_pairs is omitted, automatic port connection is attempted when there is
only a single output on the upstream component and a single input on the downstream component.

)doc")

// Interface port methods
PYDOC(add_interface_port, R"doc(
Add an interface port.

This is a convenience method that automatically sets is_input=true.

Parameters
----------
external_name : str
    The name of the interface port (used in add_flow calls)
internal_op : holoscan.core.Operator or holoscan.core.Subgraph
    The internal operator (or subgraph) that owns the actual port
internal_port : str
    The port name on the internal operator (or the interface port name on
    the internal subgraph).
is_input : bool
    Whether this is an input port (vs output)
)doc")

// Interface port methods
PYDOC(add_input_interface_port, R"doc(
Add an input interface port (convenience method).

This is a convenience method that automatically sets is_input=true.

Parameters
----------
external_name : str
    The name of the interface port (used in add_flow calls)
internal_op : holoscan.core.Operator or holoscan.core.Subgraph
    The internal operator (or subgraph) that owns the actual port
internal_port : str
    The port name on the internal operator (or the interface port name on
    the internal subgraph).
)doc")

PYDOC(add_output_interface_port, R"doc(
Add an output interface port (convenience method).

This is a convenience method that automatically sets is_input=false.

Parameters
----------
external_name : str
    The name of the interface port (used in add_flow calls)
internal_op : holoscan.core.Operator or holoscan.core.Subgraph
    The internal operator (or subgraph) that owns the actual port
internal_port : str
    The port name on the internal operator (or the interface port name on
    the internal subgraph)
)doc")

// Execution interface port methods
PYDOC(add_input_exec_interface_port, R"doc(
Add an input execution interface port.

This method exposes an internal operator's or nested subgraph's execution input port as an
external connection point for control flow.

Parameters
----------
external_name : str
    The name of the interface port (used in add_flow calls)
internal_op : Operator
    The internal operator to expose as an execution target (operator overload only)

Other Parameters
----------------
internal_subgraph : Subgraph
    The internal subgraph to expose (subgraph overload only)
internal_interface_port : str
    When using Subgraph overload, the name of the execution interface port on that Subgraph
    (subgraph overload only)

Notes
-----
This is an overloaded function with two variants:

1. Operator variant: add_input_exec_interface_port(external_name, internal_op)
   - Exposes an operator's execution input as an external interface port
   - The operator must be a Native operator (not GXF)

2. Subgraph variant: add_input_exec_interface_port(external_name, internal_subgraph, internal_interface_port)
   - Exposes a nested subgraph's execution interface port as an external interface port
   - Enables hierarchical execution control flow composition
)doc")

PYDOC(add_output_exec_interface_port, R"doc(
Add an output execution interface port.

This method exposes an internal operator's or nested subgraph's execution output port as an
external connection point for control flow.

Parameters
----------
external_name : str
    The name of the interface port (used in add_flow calls)
internal_op : Operator
    The internal operator to expose as an execution source (operator overload only)

Other Parameters
----------------
internal_subgraph : Subgraph
    The internal subgraph to expose (subgraph overload only)
internal_interface_port : str
    When using Subgraph overload, the name of the execution interface port on that subgraph
    (subgraph overload only)

Notes
-----
This is an overloaded function with two variants:

1. Operator variant:
   - ``add_output_exec_interface_port(external_name, internal_op)``
   - Exposes an operator's execution output as an external interface port
   - The operator must be a native operator (not a ``GXFOperator``)

2. Subgraph variant:
   - ``add_output_exec_interface_port(external_name, internal_subgraph, internal_interface_port)``
   - Exposes a nested subgraph's execution interface port as an external interface port
   - Enables hierarchical execution control flow composition
)doc")

}  // namespace Subgraph

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CORE_SUBGRAPH_PYDOC_HPP */
