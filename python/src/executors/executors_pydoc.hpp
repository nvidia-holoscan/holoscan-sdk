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

#ifndef PYHOLOSCAN_EXECUTORS_PYDOC_HPP
#define PYHOLOSCAN_EXECUTORS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace GXFExecutor {

// Constructor
PYDOC(GXFExecutor, R"doc(
GXF-based executor class.
)doc")

PYDOC(GXFExecutor_app, R"doc(
GXF-based executor class.

Parameters
----------
app : holoscan.core.Fragment
    The fragment associated with the executor.

)doc")

PYDOC(create_input_port, R"doc(
Create and setup GXF components for an input port.

For a given input port specification, create a GXF Receiver component for the port and
create a GXF SchedulingTerm component that is corresponding to the Condition of the port.

If there is no condition specified for the port, a default condition
(MessageAvailableCondition) is created.
It currently supports ConditionType::kMessageAvailable and ConditionType::kNone condition
types.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that this operator belongs to.
gxf_context : PyCapsule object
    The operator's associated GXF context.
eid : int
    The GXF entity ID.
io_spec : IOSpec
    Input port specification.
bind_port : bool
    If ``True``, bind the port to the existing GXF Receiver component.
    Otherwise, create a new GXF Receiver component.
)doc")

PYDOC(create_output_port, R"doc(
Create and setup GXF components for an output port.

For a given output port specification, create a GXF Transmitter component for the port and
create a GXF SchedulingTerm component that is corresponding to the Condition of the port.

If there is no condition specified for the port, a default condition
(`DownstreamMessageAffordableCondition`) is created. It currently supports
`ConditionType::kDownstreamMessageAffordable` and `ConditionType::kNone`
condition types.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that this operator belongs to.
gxf_context : PyCapsule object
    The operator's associated GXF context.
eid : int
    The GXF entity ID.
io_spec : IOSpec
    Output port specification.
bind_port : bool
    If ``True``, bind the port to the existing GXF Transmitter component.
    Otherwise, create a new GXF Transmitter component.
)doc")

}  // namespace GXFExecutor

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_EXECUTORS_PYDOC_HPP
