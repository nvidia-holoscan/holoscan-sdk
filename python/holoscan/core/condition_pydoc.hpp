/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CORE_CONDITION_PYDOC_HPP
#define PYHOLOSCAN_CORE_CONDITION_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace ConditionType {

//  Constructor
PYDOC(ConditionType, R"doc(
Enum class for Condition types.
)doc")

}  // namespace ConditionType

namespace SchedulingStatusType {

//  Constructor
PYDOC(SchedulingStatusType, R"doc(
Enum class for Condition scheduling status.
)doc")

}  // namespace SchedulingStatusType

namespace Condition {

PYDOC(Condition, R"doc(
Class representing a condition.
)doc")

//  Constructor
PYDOC(Condition_args_kwargs, R"doc(
Class representing a condition.

Can be initialized with any number of Python positional and keyword arguments.

If a `name` keyword argument is provided, it must be a `str` and will be
used to set the name of the condition.

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

PYDOC(condition_type, R"doc(
Condition type.

`holoscan.core.Condition.ConditionComponentType` enum representing the type of
the condition. The two types currently implemented are NATIVE and GXF.
)doc")

PYDOC(receiver, R"doc(
Get the receiver used by an input port of the operator this condition is associated with.

Parameters
----------
port_name : str
    The name of the input port.

Returns
-------
receiver : holoscan.resources.Receiver
    The receiver used by this input port. Will be None if the port does not exist.
)doc")

PYDOC(transmitter, R"doc(
Get the transmitter used by an output port of the operator this condition is associated with.

Parameters
----------
port_name : str
    The name of the output port.

Returns
-------
transmitter : holoscan.resources.Transmitter or None
    The transmitter used by this output port. Will be None if the port does not exist.
)doc")

}  // namespace Condition

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_CONDITION_PYDOC_HPP
