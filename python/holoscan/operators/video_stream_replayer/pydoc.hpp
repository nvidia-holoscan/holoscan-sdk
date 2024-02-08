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

#ifndef HOLOSCAN_OPERATORS_VIDEO_STREAM_REPLAYER_PYDOC_HPP
#define HOLOSCAN_OPERATORS_VIDEO_STREAM_REPLAYER_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::VideoStreamReplayerOp {

PYDOC(VideoStreamReplayerOp, R"doc(
Operator class to replay a video stream from a file.
)doc")

// PyVideoStreamReplayerOp Constructor
PYDOC(VideoStreamReplayerOp_python, R"doc(
Operator class to replay a video stream from a file.

Named output:
    output: nvidia::gxf::Tensor
        A message containing a video frame deserialized from disk.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
directory : str
    Directory path for reading files from.
basename : str
    User specified file name without extension.
batch_size : int, optional
    Number of entities to read and publish for one tick.
ignore_corrupted_entities : bool, optional
    If an entity could not be deserialized, it is ignored by default;
    otherwise a failure is generated.
frame_rate : float, optional
    Frame rate to replay. If zero value is specified, it follows timings in
    timestamps.
realtime : bool, optional
    Playback video in realtime, based on frame_rate or timestamps.
repeat : bool, optional
    Repeat video stream in a loop.
count : int, optional
    Number of frame counts to playback. If zero value is specified, it is
    ignored. If the count is less than the number of frames in the video, it
    would finish early.
name : str, optional
    The name of the operator.
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

#endif /* HOLOSCAN_OPERATORS_VIDEO_STREAM_REPLAYER_PYDOC_HPP */
