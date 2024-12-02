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

#ifndef PYHOLOSCAN_CORE_ARG_PYDOC_HPP
#define PYHOLOSCAN_CORE_ARG_PYDOC_HPP

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
Name of the argument.
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
}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_ARG_PYDOC_HPP
