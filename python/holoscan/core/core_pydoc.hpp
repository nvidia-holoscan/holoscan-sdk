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

#ifndef PYHOLOSCAN_CORE_PYDOC_HPP
#define PYHOLOSCAN_CORE_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace ArgElementType {

PYDOC(ArgElementType, R"doc(
Enum class for an `Arg`'s element type.
)doc")

}  // namespace ArgElementType
namespace ArgContainerType {

PYDOC(ArgContainerType, R"doc(
Enum class for an `Arg`'s container type.
)doc")

}  // namespace ArgContainerType

namespace Arg {

//  Constructor
PYDOC(Arg, R"doc(
Class representing a typed argument.

Parameters
----------
name : str, optional
    The argument's name.
)doc")

PYDOC(name, R"doc(
The name of the argument.

Returns
-------
name : str
)doc")

PYDOC(arg_type, R"doc(
ArgType info corresponding to the argument.

Returns
-------
arg_type : holoscan.core.ArgType
)doc")

PYDOC(has_value, R"doc(
Boolean flag indicating whether a value has been assigned to the argument.
)doc")

PYDOC(description, R"doc(
YAML formatted string describing the argument.
)doc")

}  // namespace Arg

namespace ArgList {

//  Constructor
PYDOC(ArgList, R"doc(
Class representing a list of arguments.
)doc")

PYDOC(name, R"doc(
The name of the argument list.

Returns
-------
name : str
)doc")

PYDOC(size, R"doc(
The number of arguments in the list.
)doc")

PYDOC(args, R"doc(
The underlying list of `Arg` objects.
)doc")

PYDOC(clear, R"doc(
Clear the argument list.
)doc")

PYDOC(add_Arg, R"doc(
Add an argument to the list.
)doc")

PYDOC(add_ArgList, R"doc(
Add a list of arguments to the list.
)doc")

PYDOC(description, R"doc(
YAML formatted string describing the list.
)doc")

// note: docs for overloadded add method are defined in core.cpp

}  // namespace ArgList

namespace ArgType {

PYDOC(ArgType, R"doc(
Class containing argument type info.
)doc")

//  Constructor
PYDOC(ArgType_kwargs, R"doc(
Class containing argument type info.

Parameters
----------
element_type : holoscan.core.ArgElementType
  Element type of the argument.

container_type : holoscan.core.ArgContainerType
  Container type of the argument.

)doc")

PYDOC(element_type, R"doc(
The element type of the argument.
)doc")

PYDOC(container_type, R"doc(
The container type of the argument.
)doc")

PYDOC(dimension, R"doc(
The dimension of the argument container.
)doc")

PYDOC(to_string, R"doc(
String describing the argument type.
)doc")

// note: docs for overloaded add method are defined in core.cpp
}  // namespace ArgType

namespace Component {

//  Constructor
PYDOC(Component, R"doc(
Base component class.
)doc")

PYDOC(name, R"doc(
The name of the component.

Returns
-------
name : str
)doc")

PYDOC(fragment, R"doc(
The fragment containing the component.

Returns
-------
name : holoscan.core.Fragment
)doc")

PYDOC(id, R"doc(
The identifier of the component.

The identifier is initially set to -1, and will become a valid value when the
component is initialized.

With the default executor (`holoscan.gxf.GXFExecutor`), the identifier is set to the GXF
component ID.

Returns
-------
id : int
)doc")

PYDOC(add_arg_Arg, R"doc(
Add an argument to the component.
)doc")

PYDOC(add_arg_ArgList, R"doc(
Add a list of arguments to the component.
)doc")

PYDOC(args, R"doc(
The list of arguments associated with the component.

Returns
-------
arglist : holoscan.core.ArgList
)doc")

PYDOC(initialize, R"doc(
Initialize the component.
)doc")

PYDOC(description, R"doc(
YAML formatted string describing the component.
)doc")

}  // namespace Component

namespace ConditionType {

//  Constructor
PYDOC(ConditionType, R"doc(
Enum class for Condition types.
)doc")

}  // namespace ConditionType

namespace Condition {

PYDOC(Condition, R"doc(
Class representing a condition.
)doc")

//  Constructor
PYDOC(Condition_args_kwargs, R"doc(
Class representing a condition.

Can be initialized with any number of Python positional and keyword arguments.

If a `name` keyword argument is provided, it must be a `str` and will be
used to set the name of the Operator.

If a `fragment` keyword argument is provided, it must be of type
`holoscan.core.Fragment` (or
`holoscan.core.Application`). A single `Fragment` object can also be
provided positionally instead.

Any other arguments will be cast from a Python argument type to a C++ `Arg`
and stored in ``self.args``. (For details on how the casting is done, see the
`py_object_to_arg` utility).

Parameters
----------
\*args
    Positional arguments.
\*\*kwargs
    Keyword arguments.

Raises
------
RuntimeError
    If `name` kwarg is provided, but is not of `str` type.
    If multiple arguments of type `Fragment` are provided.
    If any other arguments cannot be converted to `Arg` type via `py_object_to_arg`.
)doc")

PYDOC(name, R"doc(
The name of the condition.

Returns
-------
name : str
)doc")

PYDOC(fragment, R"doc(
Fragment that the condition belongs to.

Returns
-------
name : holoscan.core.Fragment
)doc")

PYDOC(spec, R"doc(
The condition's ComponentSpec.
)doc")

PYDOC(setup, R"doc(
setup method for the condition.
)doc")

PYDOC(initialize, R"doc(
initialization method for the condition.
)doc")

PYDOC(description, R"doc(
YAML formatted string describing the condition.
)doc")

}  // namespace Condition

namespace Resource {

PYDOC(Resource, R"doc(
Class representing a resource.
)doc")

//  Constructor
PYDOC(Resource_args_kwargs, R"doc(
Class representing a resource.

Can be initialized with any number of Python positional and keyword arguments.

If a `name` keyword argument is provided, it must be a `str` and will be
used to set the name of the Operator.

If a `fragment` keyword argument is provided, it must be of type
`holoscan.core.Fragment` (or
`holoscan.core.Application`). A single `Fragment` object can also be
provided positionally instead.

Any other arguments will be cast from a Python argument type to a C++ `Arg`
and stored in ``self.args``. (For details on how the casting is done, see the
`py_object_to_arg` utility).

Parameters
----------
\*args
    Positional arguments.
\*\*kwargs
    Keyword arguments.

Raises
------
RuntimeError
    If `name` kwarg is provided, but is not of `str` type.
    If multiple arguments of type `Fragment` are provided.
    If any other arguments cannot be converted to `Arg` type via `py_object_to_arg`.
)doc")

PYDOC(name, R"doc(
The name of the resource.

Returns
-------
name : str
)doc")

PYDOC(fragment, R"doc(
Fragment that the resource belongs to.

Returns
-------
name : holoscan.core.Fragment
)doc")

PYDOC(spec, R"doc(
The condition's ComponentSpec.
)doc")

PYDOC(setup, R"doc(
setup method for the resource.
)doc")

PYDOC(initialize, R"doc(
initialization method for the resource.
)doc")

PYDOC(description, R"doc(
YAML formatted string describing the resource.
)doc")

}  // namespace Resource

namespace Scheduler {

PYDOC(Scheduler, R"doc(
Class representing a scheduler.
)doc")

//  Constructor
PYDOC(Scheduler_args_kwargs, R"doc(
Class representing a scheduler.

Can be initialized with any number of Python positional and keyword arguments.

If a `name` keyword argument is provided, it must be a `str` and will be
used to set the name of the Operator.

If a `fragment` keyword argument is provided, it must be of type
`holoscan.core.Fragment` (or
`holoscan.core.Application`). A single `Fragment` object can also be
provided positionally instead.

Any other arguments will be cast from a Python argument type to a C++ `Arg`
and stored in ``self.args``. (For details on how the casting is done, see the
`py_object_to_arg` utility).

Parameters
----------
\*args
    Positional arguments.
\*\*kwargs
    Keyword arguments.

Raises
------
RuntimeError
    If `name` kwarg is provided, but is not of `str` type.
    If multiple arguments of type `Fragment` are provided.
    If any other arguments cannot be converted to `Arg` type via `py_object_to_arg`.
)doc")

PYDOC(name, R"doc(
The name of the scheduler.

Returns
-------
name : str
)doc")

PYDOC(fragment, R"doc(
Fragment that the scheduler belongs to.

Returns
-------
name : holoscan.core.Fragment
)doc")

PYDOC(spec, R"doc(
The scheduler's ComponentSpec.
)doc")

PYDOC(setup, R"doc(
setup method for the scheduler.
)doc")

PYDOC(initialize, R"doc(
initialization method for the scheduler.
)doc")

}  // namespace Scheduler

namespace NetworkContext {

PYDOC(NetworkContext, R"doc(
Class representing a network context.
)doc")

//  Constructor
PYDOC(NetworkContext_args_kwargs, R"doc(
Class representing a network context.

Parameters
----------
*args
    Positional arguments.
**kwargs
    Keyword arguments.

Raises
------
RuntimeError
    If `name` kwarg is provided, but is not of `str` type.
    If multiple arguments of type `Fragment` are provided.
    If any other arguments cannot be converted to `Arg` type via `py_object_to_arg`.
)doc")

PYDOC(name, R"doc(
The name of the network context.

Returns
-------
name : str
)doc")

PYDOC(fragment, R"doc(
Fragment that the network context belongs to.

Returns
-------
name : holoscan.core.Fragment
)doc")

PYDOC(spec, R"doc(
The network context's ComponentSpec.
)doc")

PYDOC(setup, R"doc(
setup method for the network context.
)doc")

PYDOC(initialize, R"doc(
initialization method for the network context.
)doc")

}  // namespace NetworkContext

namespace InputContext {

PYDOC(InputContext, R"doc(
Class representing an input context.
)doc")

}  // namespace InputContext

namespace OutputContext {

PYDOC(OutputContext, R"doc(
Class representing an output context.
)doc")

}  // namespace OutputContext

namespace ExecutionContext {

//  Constructor
PYDOC(ExecutionContext, R"doc(
Class representing an execution context.
)doc")

}  // namespace ExecutionContext

namespace Message {

PYDOC(Message, R"doc(
Class representing a message.

A message is a data structure that is used to pass data between operators.
It wraps a ``std::any`` object and provides a type-safe interface to access the data.

This class is used by the `holoscan::gxf::GXFWrapper` to support the Holoscan native operator.
The `holoscan::gxf::GXFWrapper` will hold the object of this class and delegate the message to the
Holoscan native operator.
)doc")

}  // namespace Message

namespace ComponentSpec {

//  Constructor
PYDOC(ComponentSpec, R"doc(
Component specification class.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the component belongs to.
)doc")

PYDOC(fragment, R"doc(
The fragment that the component belongs to.

Returns
-------
name : holoscan.core.Fragment
)doc")

PYDOC(params, R"doc(
The parameters associated with the component.
)doc")

PYDOC(description, R"doc(
YAML formatted string describing the component spec.
)doc")

}  // namespace ComponentSpec

namespace IOSpec {

//  Constructor
PYDOC(IOSpec, R"doc(
I/O specification class.

Parameters
----------
op_spec : holoscan.core.OperatorSpec
    Operator specification class of the associated operator.
name : str
    The name of the IOSpec object.
io_type : holoscan.core.IOSpec.IOType
    Enum indicating whether this is an input or output specification.
)doc")

PYDOC(name, R"doc(
The name of the I/O specification class.

Returns
-------
name : str
)doc")

PYDOC(io_type, R"doc(
The type (input or output) of the I/O specification class.

Returns
-------
io_type : holoscan.core.IOSpec.IOType
)doc")

PYDOC(connector_type, R"doc(
The receiver or transmitter type of the I/O specification class.

Returns
-------
connector_type : holoscan.core.IOSpec.ConnectorType
)doc")

PYDOC(conditions, R"doc(
List of Condition objects associated with this I/O specification.

Returns
-------
condition : list of holoscan.core.Condition
)doc")

PYDOC(condition, R"doc(
Add a condition to this input/output.

The following ConditionTypes are supported:

- `ConditionType.NONE`
- `ConditionType.MESSAGE_AVAILABLE`
- `ConditionType.DOWNSTREAM_MESSAGE_AFFORDABLE`
- `ConditionType.COUNT`
- `ConditionType.BOOLEAN`

Parameters
----------
kind : holoscan.core.ConditionType
    The type of the condition.
**kwargs
    Python keyword arguments that will be cast to an `ArgList` associated
    with the condition.

Returns
-------
obj : holoscan.core.IOSpec
    The self object.
)doc")

PYDOC(connector, R"doc(
Add a connector (transmitter or receiver) to this input/output.

The following ConditionTypes are supported:

- `IOSpec.ConnectorType.DEFAULT`
- `IOSpec.ConnectorType.DOUBLE_BUFFER`
- `IOSpec.ConnectorType.UCX`

If this method is not been called, the IOSpec's `connector_type` will be
`ConnectorType.DEFAULT` which will result in a DoubleBuffered receiver or
or transmitter being used (or their annotated variant if flow tracking is
enabled).

Parameters
----------
kind : holoscan.core.IOSpec.ConnectorType
    The type of the connector. For example for type `IOSpec.ConnectorType.DOUBLE_BUFFER`, a
    `holoscan.resources.DoubleBufferReceiver` will be used for an input port and a
    `holoscan.resources.DoubleBufferTransmitter` will be used for an output port.
**kwargs
    Python keyword arguments that will be cast to an `ArgList` associated
    with the resource (connector).

Returns
-------
obj : holoscan.core.IOSpec
    The self object.

Notes
-----
This is an overloaded function. Additional variants exist:

1.) A variant with no arguments will just return the `holoscan.core.Resource` corresponding to
the transmitter or receiver used by this `IOSpec` object. If None was explicitly set, it will
return ``None``.

2.) A variant that takes a single `holoscan.core.Resource` corresponding to a transmitter or
receiver as an argument. This sets the transmitter or receiver used by the `IOSpec` object.

)doc")

}  // namespace IOSpec

namespace IOType {

PYDOC(IOType, R"doc(
Enum representing the I/O specification type (input or output).
)doc")

}  // namespace IOType

namespace ConnectorType {

PYDOC(ConnectorType, R"doc(
Enum representing the receiver type (for input specs) or transmitter type (for output specs).
)doc")

}  // namespace ConnectorType

namespace ParameterFlag {

//  Constructor
PYDOC(ParameterFlag, R"doc(
Enum class for parameter flags.

The following flags are supported:
- `NONE`: The parameter is mendatory and static. It cannot be changed at runtime.
- `OPTIONAL`: The parameter is optional and might not be available at runtime.
- `DYNAMIC`: The parameter is dynamic and might change at runtime.
)doc")

}  // namespace ParameterFlag

namespace OperatorSpec {

//  Constructor
PYDOC(OperatorSpec, R"doc(
Operator specification class.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that this operator belongs to.
)doc")

PYDOC(input, R"doc(
Add an input to the specification.
)doc")

PYDOC(input_kwargs, R"doc(
Add a named input to the specification.

Parameters
----------
name : str
    The name of the input port.
)doc")

PYDOC(output, R"doc(
Add an outputput to the specification.
)doc")

PYDOC(output_kwargs, R"doc(
Add a named output to the specification.

Parameters
----------
name : str
    The name of the output port.
)doc")

PYDOC(param, R"doc(
Add a parameter to the specification.

Parameters
----------
param : name
    The name of the parameter.
default_value : object
    The default value for the parameter.

Additional Parameters
---------------------
headline : str, optional
    If provided, this is a brief "headline" description for the parameter.
description : str, optional
    If provided, this is a description for the parameter (typically more verbose than the brief
    description provided via `headline`).
kind : str, optional
    In most cases, this keyword should not be specified. If specified, the only valid option is
    currently ``kind="receivers"``, which can be used to create a parameter holding a vector of
    receivers. This effectively creates a multi-receiver input port to which any number of
    operators can be connected.
flag: holoscan.core.ParameterFlag, optional
    If provided, this is a flag that can be used to control the behavior of the parameter.
    By default, `ParameterFlag.NONE` is used.

    The following flags are supported:
    - `ParameterFlag.NONE`: The parameter is mendatory and static. It cannot be changed at runtime.
    - `ParameterFlag.OPTIONAL`: The parameter is optional and might not be available at runtime.
    - `ParameterFlag.DYNAMIC`: The parameter is dynamic and might change at runtime.

Notes
-----
This method is intended to be called within the `setup` method of an Operator.

In general, for native Python operators, it is not necessary to call `param` to register a
parameter with the class. Instead, one can just directly add parameters to the Python operator
class (e.g. For example, directly assigning ``self.param_name = value`` in __init__.py).

The one case which cannot be implemented without a call to `param` is adding a multi-receiver port
to an operator via a parameter with ``kind="receivers"`` set.
)doc")

PYDOC(description, R"doc(
YAML formatted string describing the operator spec.
)doc")

}  // namespace OperatorSpec

namespace Operator {

//  Constructor
PYDOC(Operator_args_kwargs, R"doc(
Operator class.

Can be initialized with any number of Python positional and keyword arguments.

If a `name` keyword argument is provided, it must be a `str` and will be
used to set the name of the Operator.

`Condition` classes will be added to ``self.conditions``, `Resource`
classes will be added to ``self.resources``, and any other arguments will be
cast from a Python argument type to a C++ `Arg` and stored in ``self.args``.
(For details on how the casting is done, see the `py_object_to_arg`
utility). When a Condition or Resource is provided via a kwarg, it's name
will be automatically be updated to the name of the kwarg.

Parameters
----------
fragment : holoscan.core.Fragment
    The `holoscan.core.Fragment` (or `holoscan.core.Application`) to which this Operator will
    belong.
\*args
    Positional arguments.
\*\*kwargs
    Keyword arguments.

Raises
------
RuntimeError
    If `name` kwarg is provided, but is not of `str` type.
    If multiple arguments of type `Fragment` are provided.
    If any other arguments cannot be converted to `Arg` type via `py_object_to_arg`.
)doc")

PYDOC(name, R"doc(
The name of the operator.

Returns
-------
name : str
)doc")

PYDOC(fragment, R"doc(
The fragment that the operator belongs to.

Returns
-------
name : holoscan.core.Fragment
)doc")

PYDOC(conditions, R"doc(
Conditions associated with the operator.
)doc")

PYDOC(resources, R"doc(
Resources associated with the operator.
)doc")

PYDOC(initialize, R"doc(
Operator initialization method.
)doc")

PYDOC(setup, R"doc(
Operator setup method.
)doc")

PYDOC(start, R"doc(
Operator start method.
)doc")

PYDOC(stop, R"doc(
Operator stop method.
)doc")

PYDOC(compute, R"doc(
Operator compute method. This method defines the primary computation to be
executed by the operator.
)doc")

PYDOC(operator_type, R"doc(
The operator type.

`holoscan.core.Operator.OperatorType` enum representing the type of
the operator. The two types currently implemented are native and GXF.
)doc")

PYDOC(description, R"doc(
YAML formatted string describing the operator.
)doc")

}  // namespace Operator

namespace Config {

PYDOC(Config, R"doc(
Configuration class.

Represents configuration parameters as read from a YAML file.
)doc")

//  Constructor
PYDOC(Config_kwargs, R"doc(
Configuration class.

Represents configuration parameters as read from a YAML file.

Parameters
----------
config_file : str
    The path to the configuration file (in YAML format).
prefix : str, optional
    TODO
)doc")

PYDOC(config_file, R"doc(
The configuration file (in YAML format) associated with the Config object.
)doc")

PYDOC(prefix, R"doc(
TODO
)doc")

}  // namespace Config

namespace Executor {

//  Constructor
PYDOC(Executor, R"doc(
Executor class.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the executor is associated with.
)doc")

PYDOC(fragment, R"doc(
The fragment that the executor belongs to.

Returns
-------
name : holoscan.core.Fragment
)doc")

PYDOC(run, R"doc(
Method that can be called to run the executor.
)doc")

PYDOC(context, R"doc(
The corresponding GXF context. This will be an opaque PyCapsule object.
)doc")

PYDOC(context_uint64, R"doc(
The corresponding GXF context represented as a 64-bit unsigned integer address
)doc")

}  // namespace Executor

namespace Fragment {

//  Constructor
PYDOC(Fragment, R"doc(
Fragment class.
)doc")

PYDOC(name, R"doc(
The fragment's name.

Returns
-------
name : str
)doc")

PYDOC(application, R"doc(
The application associated with the fragment.

Returns
-------
app : holoscan.core.Application
)doc")

//  Constructor
PYDOC(config_kwargs, R"doc(
Configuration class.

Represents configuration parameters as read from a YAML file.

Parameters
----------
config : str or holoscan.core.Config
    The path to the configuration file (in YAML format) or a `holoscan.core.Config`
    object.
prefix : str, optional
    Prefix path for the` config` file. Only available in the overloaded variant
    that takes a string for `config`.
)doc")

PYDOC(graph, R"doc(
Get the computation graph (Graph node is an Operator) associated with the fragment.
)doc")

PYDOC(executor, R"doc(
Get the executor associated with the fragment.
)doc")

PYDOC(from_config, R"doc(
Retrieve parameters from the associated configuration.

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
Retrieve a dictionary parameters from the associated configuration.

Parameters
----------
key : str
    The key within the configuration file to retrieve. This can also be a specific
    component of the parameter via syntax `'key.sub_key'`.

Returns
-------
kwargs : dict
    A Python dict containing the parameters in the configuration file under the
    specified key.
)doc")

PYDOC(add_operator, R"doc(
Add an operator to the fragment.

Parameters
----------
op : holoscan.core.Operator
    The operator to add.
)doc")

PYDOC(add_flow_pair, R"doc(
Connect two operators associated with the fragment.

Parameters
----------
upstream_op : holoscan.core.Operator
    Source operator.
downstream_op : holoscan.core.Operator
    Destination operator.
port_pairs : Sequence of (str, str) tuples
    Sequence of ports to connect. The first element of each 2-tuple is a port
    from `upstream_op` while the second element is the port of `downstream_op`
    to which it connects.

Notes
-----
This is an overloaded function. Additional variants exist:

1.) For the Application class there is a variant where the first two arguments are of type
`holoscan.core.Fragment` instead of `holoscan.core.Operator`. This variant is used in building
multi-fragment applications.
2.) There are also variants that omit the `port_pairs` argument that are applicable when there is
only a single output on the upstream operator/fragment and a single input on the downstream
operator/fragment.

)doc")

PYDOC(compose, R"doc(
The compose method of the Fragment.

This method should be called after `config`, but before `run` in order to
compose the computation graph.
)doc")

PYDOC(scheduler, R"doc(
Get the scheduler to be used by the Fragment.
)doc")

PYDOC(scheduler_kwargs, R"doc(
Assign a scheduler to the Fragment.

Parameters
----------
scheduler : holoscan.core.Scheduler
    A scheduler class instance to be used by the underlying GXF executor. If unspecified,
    the default is a `holoscan.gxf.GreedyScheduler`.
)doc")

PYDOC(network_context, R"doc(
Get the network context to be used by the Fragment
)doc")

PYDOC(network_context_kwargs, R"doc(
Assign a network context to the Fragment

Parameters
----------
network_context : holoscan.core.NetworkContext
    A network_context class instance to be used by the underlying GXF executor.
    If unspecified, no network context will be used.
)doc")

PYDOC(track, R"doc(
The track method of the fragment (or application).

This method enables data frame flow tracking and returns a DataFlowTracker object which can be
used to display metrics data for profiling an application.

Parameters
----------
num_start_messages_to_skip : int
    The number of messages to skip at the beginning.
num_last_messages_to_discard : int
    The number of messages to discard at the end.
latency_threshold : int
    The minimum end-to-end latency in milliseconds to account for in the
    end-to-end latency metric calculations
)doc")

PYDOC(run, R"doc(
The run method of the Fragment.

This method runs the computation. It must have first been initialized via
`config` and `compose`.
)doc")

}  // namespace Fragment

namespace Application {

// Constructor
PYDOC(Application, R"doc(
Application class.

This constructor parses the command line for flags that are recognized by App Driver/Worker,
and removes all recognized flags so users can use the remaining flags for their own purposes.

If the arguments are not specified, the arguments are retrieved from ``sys.executable`` and
``sys.argv``.

The arguments after processing arguments (parsing Holoscan-specific flags and removing them)
are accessible through the ``argv`` attribute.

Parameters
----------
argv : List[str]
    The command line arguments to parse. The first item should be the path to the python executable.
    If not specified, ``[sys.executable, *sys.argv]`` is used.

Examples
--------
>>> from holoscan.core import Application
>>> import sys
>>> Application().argv == sys.argv
True
>>> Application([]).argv == sys.argv
True
>>> Application([sys.executable, *sys.argv]).argv == sys.argv
True
>>> Application(["python3", "myapp.py", "--driver", "--address=10.0.0.1", "my_arg1"]).argv
['myapp.py', 'my_arg1']
)doc")

PYDOC(name, R"doc(
The application's name.

Returns
-------
name : str
)doc")

PYDOC(description, R"doc(
The application's description.

Returns
-------
description : str
)doc")

PYDOC(version, R"doc(
The application's version.

Returns
-------
version : str
)doc")

PYDOC(argv, R"doc(
The command line arguments after processing flags.
This does not include the python executable like `sys.argv` does.

Returns
-------
argv : list of str
)doc")

PYDOC(options, R"doc(
The reference to the CLI options.

Returns
-------
options : holoscan.core.CLIOptions
)doc")

PYDOC(fragment_graph, R"doc(
Get the computation graph (Graph node is a Fragment) associated with the application.
)doc")

PYDOC(add_operator, R"doc(
Add an operator to the application.

Parameters
----------
op : holoscan.core.Operator
    The operator to add.
)doc")

PYDOC(add_fragment, R"doc(
Add a fragment to the application.

Parameters
----------
frag : holoscan.core.Fragment
    The fragment to add.
)doc")

PYDOC(compose, R"doc(
The compose method of the application.

This method should be called after `config`, but before `run` in order to
compose the computation graph.
)doc")

PYDOC(run, R"doc(
The run method of the application.

This method runs the computation. It must have first been initialized via
`config` and `compose`.
)doc")

PYDOC(track, R"doc(
The track method of the application.

This method enables data frame flow tracking and returns
a DataFlowTracker object which can be used to display metrics data
for profiling an application.

Parameters
----------
num_start_messages_to_skip : int
    The number of messages to skip at the beginning.
num_last_messages_to_discard : int
    The number of messages to discard at the end. 
latency_threshold : int
    The minimum end-to-end latency in milliseconds to account for in the
    end-to-end latency metric calculations
)doc")
}  // namespace Application

namespace CLIOptions {

// Constructor
PYDOC(CLIOptions, R"doc(
CLIOptions class.
)doc")

PYDOC(run_driver, R"doc(
The flag to run the App Driver.
)doc")

PYDOC(run_worker, R"doc(
The flag to run the App Worker.
)doc")

PYDOC(driver_address, R"doc(
The address of the App Driver.
)doc")

PYDOC(worker_address, R"doc(
The address of the App Worker.
)doc")

PYDOC(worker_targets, R"doc(
The list of fragments for the App Worker.

Returns
-------
worker_targets : list of str
)doc")

PYDOC(config_path, R"doc(
The path to the configuration file.
)doc")

PYDOC(print, R"doc(
Print the CLI Options.
)doc")

}  // namespace CLIOptions

namespace DataFlowMetric {

//  Constructor
PYDOC(DataFlowMetric, R"doc(
Enum class for DataFlowMetric type.
)doc")

}  // namespace DataFlowMetric

namespace DataFlowTracker {

//  Constructor
PYDOC(DataFlowTracker, R"doc(
Data Flow Tracker class.

The DataFlowTracker class is used to track the data flow metrics for different paths
between the root and leaf operators. This class is used by developers to get data flow
metrics either during the execution of the application and/or as a summary after the
application ends.
)doc")

PYDOC(get_metric_with_pathstring, R"doc(
Return the value of a metric for a given path.

If `metric` is DataFlowMetric::NUM_SRC_MESSAGES, then the function returns -1.

Parameters
----------
pathstring : str
    The path name string for which the metric is being queried
metric : holoscan.core.DataFlowMetric
    The metric to be queried.

Returns
-------
val : float
    The value of the metric for the given path

Notes
-----
There is also an overloaded version of this function that takes only the `metric` argument.
)doc")

PYDOC(get_num_paths, R"doc(
The number of tracked paths

Returns
-------
num_paths : int
    The number of tracked paths
)doc")

PYDOC(get_path_strings, R"doc(
Return an array of strings which are path names. Each path name is a
comma-separated list of Operator names in a path. The paths are agnostic to the edges between
two Operators.

Returns
-------
paths : list[str]
    A list of the path names.
)doc")


PYDOC(enable_logging, R"doc(
Enable logging of frames at the end of the every execution of a leaf
Operator.

A path consisting of an array of tuples in the form of (an Operator name, message
receive timestamp, message publish timestamp) is logged in a file. The logging does not take
into account the number of message to skip or discard or the threshold latency.

This function buffers a number of lines set by `num_buffered_messages` before flushing the buffer
to the log file.

Parameters
----------
filename : str
    The name of the log file.
num_buffered_messages : int
    The number of messages to be buffered before flushing the buffer to the log file.
)doc")

PYDOC(end_logging, R"doc(
Write out any remaining messages from the log buffer and close the file
)doc")

PYDOC(print, R"doc(
Print the result of the data flow tracking in pretty-printed format to the standard output
)doc")

PYDOC(set_discard_last_messages, R"doc(
Set the number of messages to discard at the end of the execution.

This does not affect the log file or the number of source messages metric.

Parameters
----------
num : int
    The number of messages to discard.
)doc")

PYDOC(set_skip_latencies, R"doc(
Set the threshold latency for which the end-to-end latency calculations will be done.
Any latency strictly less than the threshold latency will be ignored.

This does not affect the log file or the number of source messages metric.

Parameters
----------
threshold : int
    The threshold latency in milliseconds.
)doc")

PYDOC(set_skip_starting_messages, R"doc(
Set the number of messages to skip at the beginning of the execution.

This does not affect the log file or the number of source messages metric.

Parameters
----------
num : int
    The number of messages to skip.
)doc")

}  // namespace DataFlowTracker

namespace DLDevice {

// Constructor
PYDOC(DLDevice, R"doc(
DLDevice class.
)doc")

PYDOC(device_type, R"doc(
The device type (`DLDeviceType`).

The following device types are supported:

- `DLDeviceType.DLCPU`: system memory (kDLCPU)
- `DLDeviceType.DLCUDA`: CUDA GPU memory (kDLCUDA)
- `DLDeviceType.DLCUDAHost`: CUDA pinned memory (kDLCUDAHost)
- `DLDeviceType.DLCUDAManaged`: CUDA managed memory (kDLCUDAManaged)
)doc")

PYDOC(device_id, R"doc(
The device id (int).
)doc")

}  // namespace DLDevice

namespace Tensor {

// Constructor
PYDOC(Tensor, R"doc(
Base class representing a Holoscan Tensor.
)doc")

PYDOC(as_tensor, R"doc(
Convert a Python object to a Tensor.
)doc")

PYDOC(py_dlpack, R"doc(
Convert a Tensor to __dlpack__ function.
)doc")

PYDOC(py_dlpack_device, R"doc(
Convert a Tensor to __dlpack_device__ function.
)doc")

PYDOC(ndim, R"doc(
The number of dimensions.
)doc")

PYDOC(data, R"doc(
PyCapsule handle to the tensor's data.
)doc")

PYDOC(device, R"doc(
DLDevice struct with device type/id information.
)doc")

PYDOC(dtype, R"doc(
DLDataType struct with data type information.

For details of the DLDataType struct see the DLPack documentation:
https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv410DLDataType
)doc")

PYDOC(shape, R"doc(
Tuple containing the tensor's shape.
)doc")

PYDOC(strides, R"doc(
Tuple containing the tensor's strides (in elements).
)doc")

PYDOC(size, R"doc(
The total number of elements in the tensor.
)doc")

PYDOC(itemsize, R"doc(
The size of a single element of the tensor's data.
)doc")

PYDOC(nbytes, R"doc(
The size of the tensor's data in bytes.
)doc")

PYDOC(dlpack, R"doc(
Exports the array for consumption by ``from_dlpack()`` as a DLPack capsule.

Please refer to the DLPack and Python array API standard documentation for more details:

- https://dmlc.github.io/dlpack/latest/python_spec.html
- https://data-apis.org/array-api/latest/design_topics/data_interchange.html
- https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html?highlight=__dlpack__

Parameters
----------
stream : Optional[Union[int, Any]]
    For CUDA, a Python integer representing a pointer to a stream, on devices that support streams.
    ``stream`` is provided by the consumer to the producer to instruct the producer to ensure that
    operations can safely be performed on the array
    (e.g., by inserting a dependency between streams via “wait for event”).
    The pointer must be a positive integer or `-1`. If stream is `-1`, the value may be used by the
    consumer to signal “producer must not perform any synchronization”.
    The ownership of the stream stays with the consumer.
    On CPU and other device types without streams, only None is accepted.

    - None: producer must assume the legacy default stream (default).
    - 1: the legacy default stream.
    - 2: the per-thread default stream.
    - > 2: stream number represented as a Python integer.
    - 0 is disallowed due to its ambiguity: 0 could mean either None, 1, or 2.

Returns
-------
dlpack : PyCapsule
    A PyCapsule object with the DLPack data.
)doc")

PYDOC(dlpack_device, R"doc(
Returns device type and device ID in DLPack format. Meant for use within ``from_dlpack()``.

Please refer to the DLPack and Python array API standard documentation for more details:

- https://dmlc.github.io/dlpack/latest/python_spec.html
- https://data-apis.org/array-api/latest/design_topics/data_interchange.html
- https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.array.__dlpack_device__.html

Returns
-------
device : Tuple[enum.Enum, int]
    A tuple of (device type, device id) in DLPack format.
)doc")

}  // namespace Tensor

namespace Core {

PYDOC(py_object_to_arg, R"doc(
Utility that converts a single python argument to a corresponding `Arg` type.

Parameters
----------
value : Any
    The python value to convert.

Returns
-------
obj : holoscan.core.Arg
    `Arg` class corresponding to the provided value. For example a Python float
    will become an `Arg` containing a C++ double while a list of Python ints
    would become an `Arg` corresponding to a ``std::vector<uint64_t>``.
name : str, optional
    A name to assign to the argument.
)doc")

PYDOC(kwargs_to_arglist, R"doc(
Utility that converts a set of python keyword arguments to an `ArgList`.

Parameters
----------
**kwargs
    The python keyword arguments to convert.

Returns
-------
arglist : holoscan.core.ArgList
    `ArgList` class corresponding to the provided keyword values. The argument
    names will match the keyword names. Values will be converted as for
    `py_object_to_arg`.
)doc")

PYDOC(arg_to_py_object, R"doc(
Utility that converts an `Arg` to a corresponding Python object.

Parameters
----------
arg : holoscan.core.Arg
    The argument to convert.

Returns
-------
obj : Any
    Python object corresponding to the provided argument. For example,
    an argument of any integer type will become a Python `int` while
    `std::vector<double>` would become a list of Python floats.
)doc")

PYDOC(arglist_to_kwargs, R"doc(
Utility that converts an `ArgList` to a Python kwargs dictionary.

Parameters
----------
arglist : holoscan.core.ArgList
    The argument list to convert.

Returns
-------
kwargs : dict
    Python dictionary with keys matching the names of the arguments in
    `ArgList`. The values will be converted as for `arg_to_py_object`.
)doc")

}  // namespace Core

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_PYDOC_HPP
