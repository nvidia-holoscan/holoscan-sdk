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

#ifndef PYHOLOSCAN_OPERATORS_VIDEO_STREAM_REPLAYER_PYDOC_HPP
#define PYHOLOSCAN_OPERATORS_VIDEO_STREAM_REPLAYER_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::VideoStreamReplayerOp {

// PyVideoStreamReplayerOp Constructor
PYDOC(VideoStreamReplayerOp, R"doc(
Operator class to replay a video stream from a file.

**==Named Outputs==**

    output : nvidia::gxf::Tensor
        A message containing a video frame deserialized from disk. Depending on the metadata in the
        file being read, this tensor could be on either CPU or GPU. For the data used in examples
        distributed with the SDK, the tensor will be an unnamed GPU tensor (name == "").

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the operator belongs to.
directory : str
    Directory path for reading files from.
basename : str
    User specified file name without extension.
batch_size : int, optional
    Number of entities to read and publish for one tick. Default value is ``1``.
ignore_corrupted_entities : bool, optional
    If an entity could not be deserialized, it is ignored by default;
    otherwise a failure is generated. Default value is ``True``.
frame_rate : float, optional
    Frame rate to replay. If zero value is specified, it follows timings in
    timestamps. Default value is ``0.0``.
realtime : bool, optional
    Playback video in realtime, based on frame_rate or timestamps. Default value is ``True``.
repeat : bool, optional
    Repeat video stream in a loop. Default value is ``False``.
count : int, optional
    Number of frame counts to playback. If zero value is specified, it is
    ignored. If the count is less than the number of frames in the video, it
    would finish early. Default value is ``0``.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"video_stream_replayer"``.
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

}  // namespace holoscan::doc::VideoStreamReplayerOp

#endif /* PYHOLOSCAN_OPERATORS_VIDEO_STREAM_REPLAYER_PYDOC_HPP */
