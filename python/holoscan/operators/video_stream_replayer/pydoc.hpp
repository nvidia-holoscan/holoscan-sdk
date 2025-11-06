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

**==Device Memory Requirements==**

    This operator reads data from a file to an intermediate host buffer and then transfers the data
    to the GPU. Because both host and device memory is needed, an allocator supporting both memory
    types must be used. Options for this are `UnboundedAllocator` and `RMMAllocator`. When
    specifying memory pool sizes for `RMMAllocator`, the following memory blocks are needed:

     1. One block of host memory equal in size to a single uncompressed video frame
       is needed. Note that for RMMAllocator, the memory sizes should be specified in MiB, so the
       minimum value can be obtained by
       ``math.ceil(height * width * channels * element_size_bytes) / (1024 * 1024))``

     2. One block of device memory equal in size to the host memory block.

    When declaring an `RMMAllocator` memory pool, `host_memory_initial_size` and
    `device_memory_initial_size` must be greater than or equal to the values discussed above.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph (constructor only)
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
allocator : holoscan.core.Allocator
    Allocator used to tensor memory. Currently, only the ``holoscan.resources.UnboundedAllocator``
    is supported. The default value of ``None`` will lead to use of a
    ``holoscan.resources.UnboundedAllocator``.
entity_serializer : holoscan.core.EntitySerializer
    The entity serializer used for deserialization. The default value of ``None``
    will lead to use of a default ``holoscan.resources.StdEntitySerializer``. If this argument is
    specified, then the `allocator` argument is ignored.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"video_stream_replayer"``.
)doc")

}  // namespace holoscan::doc::VideoStreamReplayerOp

#endif /* PYHOLOSCAN_OPERATORS_VIDEO_STREAM_REPLAYER_PYDOC_HPP */
