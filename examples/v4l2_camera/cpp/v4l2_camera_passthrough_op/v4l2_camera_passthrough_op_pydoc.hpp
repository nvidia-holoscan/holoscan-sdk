/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef v4l2_camera_passthrough_op_PYDOC_HPP
#define v4l2_camera_passthrough_op_PYDOC_HPP

#include <string>

// Define PYDOC macro if not already defined
#ifndef PYDOC
#define PYDOC(name, doc) constexpr auto doc_##name = doc
#endif

namespace holoscan::doc::V4L2CameraPassthroughOp {

// PyV4L2CameraPassthroughOp Constructor
PYDOC(V4L2CameraPassthroughOp, R"doc(
V4L2 Camera Passthrough operator.

This is a simple passthrough operator that receives an input entity (which may
contain either a Tensor or VideoBuffer component) and emits it on its output port.
This operator is intended for use in the v4l2_camera example application
to work around two issues:
1. Connecting multiple operators to HolovizOp "receivers" would create multiple
   input ports with MessageAvailableConditions, resulting in a deadlock
2. Pure Python operators cannot forward GXF::VideoBuffer objects

**==Named Inputs==**

    input : nvidia::gxf::Entity
        The input entity to passthrough. The entity may contain nvidia::gxf::Tensor
        or nvidia::gxf::VideoBuffer components.

**==Named Outputs==**

    output : nvidia::gxf::Entity
        The same entity that was received on the input port.

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the operator belongs to.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"v4l2_camera_passthrough"``.
)doc");

}  // namespace holoscan::doc::V4L2CameraPassthroughOp

#endif /* v4l2_camera_passthrough_op_PYDOC_HPP */
