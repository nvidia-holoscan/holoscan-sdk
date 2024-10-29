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

#ifndef HOLOSCAN_OPERATORS_VIDEO_STREAM_REPLAYER_VIDEO_STREAM_REPLAYER_HPP
#define HOLOSCAN_OPERATORS_VIDEO_STREAM_REPLAYER_VIDEO_STREAM_REPLAYER_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gxf/serialization/file_stream.hpp"
#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to replay a video stream from a file.
 *
 * ==Named Outputs==
 *
 * - **output** : `nvidia::gxf::Tensor`
 *   - A message containing a video frame deserialized from disk. Depending on the metadata in the
 *     file being read, this tensor could be on either CPU or GPU. For the data used in examples
 *     distributed with the SDK, the tensor will be an unnamed GPU tensor (name == "").
 *
 * ==Parameters==
 *
 * - **directory**: Directory path for reading files from.
 * - **basename**: User specified file name without extension.
 * - **batch_size**: Number of entities to read and publish for one tick. Optional (default: `1`).
 * - **ignore_corrupted_entities**: If an entity could not be deserialized, it is ignored by
 *   default; otherwise a failure is generated. Optional (default: `true`).
 * - **frame_rate**: Frame rate to replay. If zero value is specified, it follows timings in
 *   timestamps. Optional (default: `0.0`).
 * - **realtime**: Playback video in realtime, based on frame_rate or timestamps.
 *   Optional (default: `true`).
 * - **repeat**: Repeat video stream in a loop. Optional (default: `false`).
 * - **count**: Number of frame counts to playback. If zero value is specified, it is ignored.
 *   If the count is less than the number of frames in the video, it would finish early.
 *   Optional (default: `0`).
 * - **allocator**: The allocator used for Tensor objects. Currently this can only use the default
 *   allocator type of `holoscan::UnboundedAllocator`.
 *   Optional (default: `holoscan::UnboundedAllocator`)
 * - **entity_serializer**: The entity serializer used for deserialization. The default is to use
 *   a default-initialized ``holoscan::gxzf::StdEntitySerializer``. If this argument is
 *   specified, then the `allocator` argument is ignored.
 *
 * ==Device Memory Requirements==
 *
 * This operator reads data from a file to an intermediate host buffer and then transfers the data
 * to the GPU. Because both host and device memory is needed, an allocator supporting both memory
 * types must be used. Options for this are `UnboundedAllocator` and the `RMMAllocator`. When using
 * RMMAllocator, the following memory blocks are needed:
 *  1. One block of host memory equal in size to a single uncompressed video frame
 *    is needed. Note that for RMMAllocator, the memory sizes should be specified in MiB, so the
 *    minimum value can be obtained by:
 *
 * ```cpp
 * #include <cmath>
 *
 * ceil(static_cast<double>(height * width * channels * element_size_bytes) / (1024 * 1024));
 * ```
 *
 *  2. One block of device memory equal in size to the host memory block.
 *
 * When declaring an RMMAllocator memory pool, `host_memory_initial_size` and
 * `device_memory_initial_size` must be greater than or equal to the values discussed above.
 */
class VideoStreamReplayerOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VideoStreamReplayerOp)

  VideoStreamReplayerOp() = default;

  ~VideoStreamReplayerOp() override;

  void setup(OperatorSpec& spec) override;

  void initialize() override;
  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<holoscan::IOSpec*> transmitter_;
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<std::shared_ptr<holoscan::Resource>> entity_serializer_;
  Parameter<std::shared_ptr<BooleanCondition>> boolean_scheduling_term_;
  Parameter<std::string> directory_;
  Parameter<std::string> basename_;
  Parameter<size_t> batch_size_;
  Parameter<bool> ignore_corrupted_entities_;
  Parameter<float> frame_rate_;
  Parameter<bool> realtime_;
  Parameter<bool> repeat_;
  Parameter<uint64_t> count_;

  // Internal state
  // File stream for entities
  nvidia::gxf::FileStream entity_file_stream_;
  // File stream for index
  nvidia::gxf::FileStream index_file_stream_;

  uint64_t playback_index_ = 0;
  uint64_t playback_count_ = 0;
  uint64_t index_start_timestamp_ = 0;
  uint64_t index_last_timestamp_ = 0;
  uint64_t index_timestamp_duration_ = 0;
  uint64_t index_frame_count_ = 1;
  uint64_t playback_start_timestamp_ = 0;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_VIDEO_STREAM_REPLAYER_VIDEO_STREAM_REPLAYER_HPP */
