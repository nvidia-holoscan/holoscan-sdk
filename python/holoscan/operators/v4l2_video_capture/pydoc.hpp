/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_OPERATORS_V4L2_VIDEO_CAPTURE_PYDOC_HPP
#define PYHOLOSCAN_OPERATORS_V4L2_VIDEO_CAPTURE_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::V4L2VideoCaptureOp {

// V4L2VideoCaptureOp Constructor
PYDOC(V4L2VideoCaptureOp, R"doc(
Operator to get a video stream from a V4L2 source.

https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/v4l2.html

Inputs a video stream from a V4L2 node, including USB cameras and HDMI IN.
If no pixel format is specified in the yaml configuration file, the pixel format will be
automatically selected.
If a pixel format is specified in the yaml file, then this format will be used.

**==Named Outputs==**

    signal : nvidia::gxf::VideoBuffer or nvidia::gxf::Tensor
        A message containing a video buffer if the V4L2 pixel format has equivalent
        ``nvidia::gxf::VideoFormat``, else a tensor.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph (constructor only)
    The fragment that the operator belongs to.
allocator : holoscan.resources.Allocator
    Deprecated, do not use.
device : str
    The device to target (e.g. "/dev/video0" for device 0). Default value is ``"/dev/video0"``.
width : int, optional
    Width of the video stream. If set to ``0``, use the default width of the device. Default value
    is ``0``.
height : int, optional
    Height of the video stream. If set to ``0``, use the default height of the device. Default value
    is ``0``.
frame_rate: float, optional
    Frame rate of the video stream. If the device does not support the exact frame rate, the nearest
    match is used instead. If set to ``0``, use the default frame rate of the device. Default value
    is ``0.0``.
num_buffers : int, optional
    Number of V4L2 buffers to use. Default value is ``4``.
pixel_format : str
    Video stream pixel format (little endian four character code (fourcc)).
    Default value is ``"auto"``.
pass_through : bool
    Deprecated, do not use.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"v4l2_video_capture"``.
exposure_time : int, optional
    Exposure time of the camera sensor in multiples of 100 μs (e.g. setting exposure_time to 100 is
    10 ms).
    Default: auto exposure, or camera sensor default.
    Use `v4l2-ctl -d /dev/<your_device> -L` for a range of values supported by your device.
    When not set by the user, V4L2_CID_EXPOSURE_AUTO is set to V4L2_EXPOSURE_AUTO, or to
    V4L2_EXPOSURE_APERTURE_PRIORITY if the former is not supported.
    When set by the user, V4L2_CID_EXPOSURE_AUTO is set to V4L2_EXPOSURE_SHUTTER_PRIORITY, or to
    V4L2_EXPOSURE_MANUAL if the former is not supported. The provided value is then used to set
    V4L2_CID_EXPOSURE_ABSOLUTE.
gain : int, optional
    Gain of the camera sensor.
    Default: auto gain, or camera sensor default.
    Use `v4l2-ctl -d /dev/<your_device> -L` for a range of values supported by your device.
    When not set by the user, V4L2_CID_AUTOGAIN is set to true (if supported).
    When set by the user, V4L2_CID_AUTOGAIN is set to false (if supported). The provided value is
    then used to set V4L2_CID_GAIN.

Metadata
--------
V4L2_pixel_format : str
    V4L2 pixel format.
V4L2_ycbcr_encoding : str
    V4L2 YCbCr encoding (``enum v4l2_ycbcr_encoding`` value as string).
V4L2_quantization : str
    V4L2 quantization (``enum v4l2_quantization`` value as string).
)doc")

}  // namespace holoscan::doc::V4L2VideoCaptureOp

#endif /* PYHOLOSCAN_OPERATORS_V4L2_VIDEO_CAPTURE_PYDOC_HPP */
