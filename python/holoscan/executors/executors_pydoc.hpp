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

}  // namespace GXFExecutor

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_EXECUTORS_PYDOC_HPP
