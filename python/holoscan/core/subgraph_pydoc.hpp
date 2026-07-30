/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

PYDOC(add_subgraph, R"doc(
Add a pre-constructed subgraph as a nested subgraph, taking ownership.

This method stores the subgraph for lifetime management and interface port resolution.
The subgraph's name must already be qualified with this parent subgraph's name prefix
(which happens automatically when the subgraph is constructed with this subgraph as
the parent argument).

Parameters
----------
subgraph : Subgraph
    The subgraph to add.
)doc")

// Configuration methods
PYDOC(config_kwargs, R"doc(
Get the configuration object for this subgraph.

Returns the `holoscan.core.Config` object containing the configuration parameters
that were loaded from a YAML file passed to the subgraph constructor.

.. note::

    Configuration must be passed to the subgraph constructor via the ``config`` parameter.
    It cannot be set after construction.

.. note::

    Loading GXF extensions is not supported from the subgraph config file. GXF extensions
    should be loaded via the application-level configuration only.

Returns
-------
holoscan.core.Config
    The configuration object for this subgraph.
)doc")

PYDOC(config_keys, R"doc(
The set of keys present in the subgraph's configuration file.
)doc")

PYDOC(from_config, R"doc(
Retrieve parameters from the subgraph's associated configuration.

Parameters
----------
key : str
    The key within the configuration file to retrieve. This can also be a specific
    component of the parameter via syntax `'key.sub_key'`.

Returns
-------
args : holoscan.core.ArgList
    An argument list associated with the key.
)doc")

PYDOC(kwargs, R"doc(
Retrieve a dictionary of parameters from the subgraph's associated configuration.

This is the Python equivalent for passing configuration to operators. Use with
``**`` unpacking to pass parameters to operator constructors.

Parameters
----------
key : str
    The key within the configuration file to retrieve. This can also be a specific
    component of the parameter via syntax `'key.sub_key'`.

Returns
-------
dict
    A dictionary of the parameters stored under key.

Examples
--------
.. code-block:: python

    def compose(self):
        op = MyOp(self, name="my_op", **self.kwargs("my_op"))
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

If is_input is not specified, the port direction is auto-detected:
- For operator ports: checks if the port exists as an input or output
- For subgraph ports: uses the nested subgraph's interface port direction

If the port name exists as both an input and output (rare), you must
either specify the is_input argument explicitly, or use
add_input_interface_port() or add_output_interface_port() instead.

Parameters
----------
external_name : str
    The name of the interface port (used in add_flow calls)
internal_op : holoscan.core.Operator or holoscan.core.Subgraph
    The internal operator (or subgraph) that owns the actual port
internal_port : str, optional
    The port name on the internal operator (or the interface port name on
    the internal subgraph). Defaults to external_name if not specified.
is_input : bool, optional
    Whether this is an input port (vs output). If not specified, auto-detected
    from the operator or subgraph's port definitions.
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

PYDOC(add_data_logger, R"doc(
Add a data logger to the fragment.

This method dispatches to the fragment's add_data_logger method.

Parameters
----------
logger : holoscan.core.DataLogger
    The data logger to add
)doc")

PYDOC(interface_ports, R"doc(
Get all data interface ports.

Returns a dictionary mapping interface port names to InterfacePort objects.
Each InterfacePort contains a list of mappings - most ports have a single mapping,
but input interface ports can have multiple mappings to broadcast to multiple
internal operators.

Returns
-------
dict
    A dictionary mapping interface port names to InterfacePort objects.
)doc")

PYDOC(exec_interface_ports, R"doc(
Get all execution interface ports.

Returns
-------
dict
    A dictionary mapping execution interface port names to InterfacePort objects.
)doc")

PYDOC(get_interface_operator_port, R"doc(
Get the first operator and port name for a data interface port.

Resolves the interface port name to the actual internal operator and port,
recursively checking nested subgraphs for hierarchical port resolution.

For broadcast input ports that have multiple mappings, this returns
only the first mapping. Access the InterfacePort directly via ``interface_ports()``
to get all mappings via its ``mappings`` attribute.

Parameters
----------
port_name : str
    The interface port name to resolve.

Returns
-------
tuple of (Operator, str)
    A tuple of (operator, port_name) if found, or (None, "") if not found.
)doc")

PYDOC(get_exec_interface_operator_port, R"doc(
Get the operator and port name for an execution interface port.

Resolves the execution interface port name to the actual internal operator and port,
recursively checking nested subgraphs for hierarchical port resolution.

Parameters
----------
port_name : str
    The execution interface port name to resolve.

Returns
-------
tuple of (Operator, str)
    A tuple of (operator, port_name) if found, or (None, "") if not found.
)doc")

PYDOC(operators, R"doc(
Get all operators belonging to this subgraph and its nested subgraphs.

Returns all operators whose names are prefixed with this subgraph's name
followed by an underscore. This includes operators from nested subgraphs
since their names are also prefixed with the parent subgraph's name.

Returns
-------
list of Operator
    List of operators belonging to this subgraph.
)doc")

PYDOC(nested_subgraphs, R"doc(
Get the nested subgraphs directly owned by this subgraph.

Returns subgraphs added via ``make_subgraph`` (C++) or the ``Subgraph`` constructor (Python),
as well as subgraphs added via ``add_subgraph``. Does not recursively include subgraphs
nested further down the hierarchy.

Returns
-------
list of Subgraph
    The direct child subgraphs of this subgraph.
)doc")

}  // namespace Subgraph

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CORE_SUBGRAPH_PYDOC_HPP */
