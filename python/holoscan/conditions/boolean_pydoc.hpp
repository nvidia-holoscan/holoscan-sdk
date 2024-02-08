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

#ifndef PYHOLOSCAN_CONDITIONS_BOOLEAN_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_BOOLEAN_PYDOC_HPP

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
}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CONDITIONS_BOOLEAN_PYDOC_HPP
