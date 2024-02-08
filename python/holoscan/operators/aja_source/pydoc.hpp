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

#ifndef HOLOSCAN_OPERATORS_AJA_SOURCE_PYDOC_HPP
#define HOLOSCAN_OPERATORS_AJA_SOURCE_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::AJASourceOp {

PYDOC(AJASourceOp, R"doc(
Operator to get a video stream from an AJA capture card.
)doc")

// PyAJASourceOp Constructor
PYDOC(AJASourceOp_python, R"doc(
Operator to get a video stream from an AJA capture card.

Named inputs:
    overlay_buffer_input: nvidia::gxf::VideoBuffer (optional)
        The operator does not require a message on this input port in order for ``compute`` to be
        called. If a message is found, and ``enable_overlay`` is ``True``, the image will be mixed
        with the image captured by the AJA card. If `enable_overlay` is ``False``, any message on
        this port will be ignored.

Named outputs:
    video_buffer_output: nvidia::gxf::VideoBuffer
        The output video frame from the AJA capture card. If ``overlay_rdma`` is ``True``, this
        video buffer will be on the device, otherwise it will be in pinned host memory.
    overlay_buffer_output: nvidia::gxf::VideoBuffer (optional)
        This output port will only emit a video buffer when ``enable_overlay`` is ``True``. If
        ``overlay_rdma`` is ``True``, this video buffer will be on the device, otherwise it will be
        in pinned host memory.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
device : str, optional
    The device to target (e.g. "0" for device 0)
channel : holoscan.operators.NTV2Channel or int, optional
    The camera NTV2Channel to use for output.
width : int, optional
    Width of the video stream.
height : int, optional
    Height of the video stream.
framerate : int, optional
    Frame rate of the video stream.
rdma : bool, optional
    Boolean indicating whether RDMA is enabled.
enable_overlay : bool, optional
    Boolean indicating whether a separate overlay channel is enabled.
overlay_channel : holoscan.operators.NTV2Channel or int, optional
    The camera NTV2Channel to use for overlay output.
overlay_rdma : bool, optional
    Boolean indicating whether RDMA is enabled for the overlay.
name : str, optional
    The name of the operator.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

}  // namespace holoscan::doc::AJASourceOp

#endif /* HOLOSCAN_OPERATORS_AJA_SOURCE_PYDOC_HPP */
