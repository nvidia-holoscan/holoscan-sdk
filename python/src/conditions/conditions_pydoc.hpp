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

#ifndef PYHOLOSCAN_CONDITIONS_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace BooleanCondition {

PYDOC(BooleanCondition, R"doc(
Boolean condition class.

Used to control whether an entity is executed.
)doc")

// PyBooleanCondition Constructor
PYDOC(BooleanCondition_python, R"doc(
Boolean condition.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment the condition will be associated with
enable_tick : bool, optional
    Boolean value for the condition.
name : str, optional
    The name of the condition.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the condition.

Returns
-------
str
    The GXF type name of the condition
)doc")

PYDOC(enable_tick, R"doc(
Set condition to ``True``.
)doc")

PYDOC(disable_tick, R"doc(
Set condition to ``False``.
)doc")

PYDOC(check_tick_enabled, R"doc(
Check whether the condition is ``True``.
)doc")

PYDOC(setup, R"doc(
Define the component specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    Component specification associated with the condition.
)doc")

}  // namespace BooleanCondition

namespace CountCondition {

PYDOC(CountCondition, R"doc(
Count condition class.
)doc")

// PyCountCondition Constructor
PYDOC(CountCondition_python, R"doc(
Count condition.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment the condition will be associated with
count : int
    The execution count value used by the condition.
name : str, optional
    The name of the condition.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the condition.

Returns
-------
str
    The GXF type name of the condition
)doc")

PYDOC(count, R"doc(
The execution count associated with the condition
)doc")

PYDOC(setup, R"doc(
Define the component specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    Component specification associated with the condition.
)doc")

}  // namespace CountCondition

namespace DownstreamMessageAffordableCondition {

PYDOC(DownstreamMessageAffordableCondition, R"doc(
Condition that permits execution when the downstream operator can accept new messages.
)doc")

// PyDownstreamMessageAffordableCondition Constructor
PYDOC(DownstreamMessageAffordableCondition_python, R"doc(
Condition that permits execution when the downstream operator can accept new messages.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment the condition will be associated with
min_size : int
    The minimum number of free slots present in the back buffer.
name : str, optional
    The name of the condition.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the condition.

Returns
-------
str
    The GXF type name of the condition
)doc")

PYDOC(setup, R"doc(
Define the component specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    Component specification associated with the condition.
)doc")

PYDOC(transmitter, R"doc(
The transmitter associated with the condition.
)doc")

PYDOC(min_size, R"doc(
The minimum number of free slots required for the downstream entity's back
buffer.
)doc")

PYDOC(initialize, R"doc(
Initialize the condition

This method is called only once when the condition is created for the first
time, and uses a light-weight initialization.
)doc")

}  // namespace DownstreamMessageAffordableCondition

namespace MessageAvailableCondition {

PYDOC(MessageAvailableCondition, R"doc(
Condition that permits execution when an upstream message is available.

Executed when the associated receiver queue has at least a certain number of
elements. The receiver is specified using the receiver parameter of the
scheduling term. The minimum number of messages that permits the execution of
the entity is specified by `min_size`. An optional parameter for this
scheduling term is `front_stage_max_size`, the maximum front stage message
count. If this parameter is set, the scheduling term will only allow execution
if the number of messages in the queue does not exceed this count. It can be
used for operators which do not consume all messages from the queue.
)doc")

// PyMessageAvailableCondition Constructor
PYDOC(MessageAvailableCondition_python, R"doc(
Condition that permits execution when an upstream message is available.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment the condition will be associated with
min_size : int
    The total number of messages over a set of input channels needed
    to permit execution.
front_stage_max_size : int
    Threshold for the number of front stage messages. Execution is only
    allowed if the number of front stage messages does not exceed this
    count.
name : str, optional
    The name of the condition.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the condition.

Returns
-------
str
    The GXF type name of the condition
)doc")

PYDOC(setup, R"doc(
Define the component specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    Component specification associated with the condition.
)doc")

PYDOC(receiver, R"doc(
The receiver associated with the condition.
)doc")

PYDOC(min_size, R"doc(
The total number of messages over a set of input channels needed
to permit execution.
)doc")

PYDOC(front_stage_max_size, R"doc(
Threshold for the number of front stage messages. Execution is only allowed
if the number of front stage messages does not exceed this count.
)doc")

PYDOC(initialize, R"doc(
Initialize the condition

This method is called only once when the condition is created for the first
time, and uses a light-weight initialization.
)doc")

}  // namespace MessageAvailableCondition
}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CONDITIONS_PYDOC_HPP
