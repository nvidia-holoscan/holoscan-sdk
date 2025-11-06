/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CONDITIONS_CUDA_BUFFER_AVAILABLE_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_CUDA_BUFFER_AVAILABLE_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace CudaBufferAvailableCondition {

PYDOC(CudaBufferAvailableCondition, R"doc(
Condition based on data availability in a CudaBuffer object.

A component which specifies the availability of data at the receiver based on the CudaBuffers
present in incoming messages.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment (or subgraph) the condition will be associated with
receiver : str, optional
    The name of the operator's input port to which the condition would apply.
name : str, optional
    The name of the condition.

Notes
-----
The `nvidia::gxf::CudaBuffer` class is currently unused by Holoscan SDK. This condition is
intended exclusively for interoperation with wrapped GXF Codelets that use GXF's CudaBuffer type.
)doc")

PYDOC(receiver, R"doc(
The receiver associated with the condition.
)doc")

}  // namespace CudaBufferAvailableCondition

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CONDITIONS_CUDA_BUFFER_AVAILABLE_PYDOC_HPP */
