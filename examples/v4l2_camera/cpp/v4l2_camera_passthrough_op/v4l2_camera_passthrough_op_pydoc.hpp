/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
