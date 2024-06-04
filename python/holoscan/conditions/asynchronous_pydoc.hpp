/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CONDITIONS_ASYNCHRONOUS_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_ASYNCHRONOUS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace AsynchronousCondition {

PYDOC(AsynchronousCondition, R"doc(
Asynchronous condition class.

Used to control whether an entity is executed.
)doc")

// PyAsynchronousCondition Constructor
PYDOC(AsynchronousCondition_python, R"doc(
Asynchronous condition.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment the condition will be associated with
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

PYDOC(event_state, R"doc(
Event state property

- AsynchronousEventState.READY
- AsynchronousEventState.WAIT
- AsynchronousEventState.EVENT_WAITING
- AsynchronousEventState.EVENT_DONE
- AsynchronousEventState.EVENT_NEVER
)doc")

PYDOC(setup, R"doc(
Define the component specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    Component specification associated with the condition.
)doc")

}  // namespace AsynchronousCondition

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CONDITIONS_ASYNCHRONOUS_PYDOC_HPP
