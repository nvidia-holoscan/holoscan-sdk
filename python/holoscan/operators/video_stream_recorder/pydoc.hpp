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

#ifndef HOLOSCAN_OPERATORS_VIDEO_STREAM_RECORDER_PYDOC_HPP
#define HOLOSCAN_OPERATORS_VIDEO_STREAM_RECORDER_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::VideoStreamRecorderOp {

// PyVideoStreamRecorderOp Constructor
PYDOC(VideoStreamRecorderOp, R"doc(
Operator class to record a video stream to a file.

**==Named Inputs==**

    input : nvidia::gxf::Tensor
        A message containing a video frame to serialize to disk. The input tensor can be on either
        the CPU or GPU. This data location will be recorded as part of the metadata serialized to
        disk and if the data is later read back in via `VideoStreamReplayerOp`, the tensor output
        of that operator will be on the same device (CPU or GPU).

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the operator belongs to.
directory : str
    Directory path for storing files.
basename : str
    User specified file name without extension.
flush_on_tick : bool, optional
    Flushes output buffer on every tick when ``True``. Default value is ``False``.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"video_stream_recorder"``.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace holoscan::doc::VideoStreamRecorderOp

#endif /* HOLOSCAN_OPERATORS_VIDEO_STREAM_RECORDER_PYDOC_HPP */
