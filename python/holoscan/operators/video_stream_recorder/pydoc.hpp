/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_OPERATORS_VIDEO_STREAM_RECORDER_PYDOC_HPP
#define PYHOLOSCAN_OPERATORS_VIDEO_STREAM_RECORDER_PYDOC_HPP

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
fragment : holoscan.core.Fragment or holoscan.core.Subgraph (constructor only)
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

}  // namespace holoscan::doc::VideoStreamRecorderOp

#endif /* PYHOLOSCAN_OPERATORS_VIDEO_STREAM_RECORDER_PYDOC_HPP */
