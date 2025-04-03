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

#ifndef PYHOLOSCAN_CORE_EXECUTOR_PYDOC_HPP
#define PYHOLOSCAN_CORE_EXECUTOR_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

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

PYDOC(interrupt, R"doc(
Interrupt execution of the application.
)doc")

PYDOC(context, R"doc(
The corresponding GXF context. This will be an opaque PyCapsule object.
)doc")

PYDOC(context_uint64, R"doc(
The corresponding GXF context represented as a 64-bit unsigned integer address
)doc")

}  // namespace Executor

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_EXECUTOR_PYDOC_HPP
