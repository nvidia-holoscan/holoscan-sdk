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

#ifndef PYHOLOSCAN_CONDITIONS_CUDA_EVENT_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_CUDA_EVENT_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace CudaEventCondition {

PYDOC(CudaEventCondition, R"doc(
Condition class to indicate data availability on CUDA stream completion via an event.

A condition which specifies the availability of data at the receiver on completion of the work on
the provided cuda stream with the help of cuda event. This condition will keep polling on the
event provided to check for data availability for consumption.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment (or subgraph) the condition will be associated with
event_name : event, optional
    The event name on which the cudaEventQuery API is called to get the status.
receiver : str, optional
    The name of the operator's input port to which the condition would apply.
name : str, optional
    The name of the condition.

Notes
-----
The `nvidia::gxf::CudaEvent` class is currently unused by Holoscan SDK. This condition is
intended exclusively for interoperation with wrapped GXF Codelets that use GXF's CudaEvent type.
)doc")

PYDOC(receiver, R"doc(
The receiver associated with the condition.
)doc")

}  // namespace CudaEventCondition

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CONDITIONS_CUDA_EVENT_PYDOC_HPP */
