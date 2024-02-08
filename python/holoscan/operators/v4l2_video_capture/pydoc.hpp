/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOHUB_OPERATORS_V4L2_VIDEO_CAPTURE_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_V4L2_VIDEO_CAPTURE_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::V4L2VideoCaptureOp {

// Constructor
PYDOC(V4L2VideoCaptureOp, R"doc(
Operator to get a video stream from a V4L2 source.
)doc")

// V4L2VideoCaptureOp Constructor
PYDOC(V4L2VideoCaptureOp_python, R"doc(
Operator to get a video stream from a V4L2 source (e.g. Built-in HDMI capture card or USB camera)

https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/v4l2.html

Inputs a video stream from a V4L2 node, including USB cameras and HDMI IN.
 - Input stream is on host. If no pixel format is specified in the yaml configuration file, the
   pixel format will be automatically selected. However, only `AB24` and `YUYV` are then supported.
   If a pixel format is specified in the yaml file, then this format will be used. However, note
   that the operator then expects that this format can be encoded as RGBA32. If not, the behaviour
   is undefined.
 - Output stream is on host. Always RGBA32 at this time.

Use `holoscan.operators.FormatConverterOp` to move data from the host to a GPU device.

Named output:
    signal: nvidia::gxf::VideoBuffer
        Emits a message containing a video buffer on the host with format GXF_VIDEO_FORMAT_RGBA.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
allocator : ``holoscan.resources.Allocator``
    Memory allocator to use for the output.
device : str
    The device to target (e.g. "/dev/video0" for device 0)
width : int, optional
    Width of the video stream.
height : int, optional
    Height of the video stream.
num_buffers : int, optional
    Number of V4L2 buffers to use.
pixel_format : str
    Video stream pixel format (little endian four character code (fourcc))
name : str, optional
    The name of the operator.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : ``holoscan.core.OperatorSpec``
    The operator specification.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

}  // namespace holoscan::doc::V4L2VideoCaptureOp

#endif  // PYHOLOHUB_OPERATORS_V4L2_VIDEO_CAPTURE_PYDOC_HPP
