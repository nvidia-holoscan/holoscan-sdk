/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_COMPONENT_PYDOC_HPP
