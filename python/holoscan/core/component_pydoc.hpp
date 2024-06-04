/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CORE_COMPONENT_PYDOC_HPP
#define PYHOLOSCAN_CORE_COMPONENT_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

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

}  // namespace ComponentSpec

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

The identifier is initially set to ``-1``, and will become a valid value when the
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

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_COMPONENT_PYDOC_HPP
