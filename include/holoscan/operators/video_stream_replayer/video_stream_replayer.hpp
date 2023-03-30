/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_STREAM_PLAYBACK_VIDEO_STREAM_REPLAYER_HPP
#define HOLOSCAN_OPERATORS_STREAM_PLAYBACK_VIDEO_STREAM_REPLAYER_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/gxf/gxf_operator.hpp"
#include "gxf/serialization/file_stream.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to replay a video stream from a file.
 */
class VideoStreamReplayerOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VideoStreamReplayerOp)

  VideoStreamReplayerOp() = default;

  ~VideoStreamReplayerOp() override;

  void setup(OperatorSpec& spec) override;

  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<holoscan::IOSpec*> transmitter_;
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

#endif /* HOLOSCAN_OPERATORS_STREAM_PLAYBACK_VIDEO_STREAM_REPLAYER_HPP */
