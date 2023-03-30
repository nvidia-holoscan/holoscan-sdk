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
The resources's ComponentSpec.
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

PYDOC(resource, R"doc(
Resource class associated with this I/O specification.

Returns
-------
resource : holoscan.core.Resource
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

}  // namespace IOSpec

namespace IOType {

PYDOC(IOType, R"doc(
Enum representing the I/O specification type (input or output).
)doc")
}  // namespace IOType

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

PYDOC(input_name, R"doc(
Add a named input to the specification.

Parameters
----------
name : str
    The name of the input port.
)doc")

PYDOC(output, R"doc(
Add an outputput to the specification.
)doc")

PYDOC(output_name, R"doc(
Add a named output to the specification.

Parameters
----------
name : str
    The name of the output port.
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

PYDOC(config_kwargs, R"doc(
Set the configuration file associated with the fragment.

Parameters
----------
config_file : str
    The path to the configuration file (in YAML format).
prefix : str, optional
    TODO
)doc")

PYDOC(config, R"doc(
Get the configuration associated with the fragment.

Returns
-------
config : holoscan.core.Config
)doc")

PYDOC(graph, R"doc(
Get the computation graph associated with the fragment.
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
port_pairs : Sequence of 2-tuples
    Sequence of ports to connect. The first element of each 2-tuple is a port
    from `upstream_op` while the second element is the port of `downstream_op`
    to which it connects.
)doc")

PYDOC(compose, R"doc(
The compose method of the Fragment.

This method should be called after `config`, but before `run` in order to
compose the computation graph.
)doc")

PYDOC(run, R"doc(
The run method of the Fragment.

This method runs the computation. It must have first been initialized via
`config` and `compose`.
)doc")

}  // namespace Fragment

namespace Application {

//  Constructor
PYDOC(Application, R"doc(
Application class.
)doc")

PYDOC(name, R"doc(
The application's name.

Returns
-------
name : str
)doc")

PYDOC(add_operator, R"doc(
Add an operator to the application.

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

}  // namespace Application

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
